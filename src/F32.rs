use ndarray::{Array2, s};

pub use self::{
    dct::dct_2d_axis,
    filter::filter,
    framing::framing,
    gammatone::fft2gammatonemx,
    mean_power_normalization::mean_power_normalization,
    medium_time_processing::{
        asymmetric_lowpass_filtering, medium_time_power_calculation,
        switch_excitation_or_non_excitation, temporal_masking, weight_smoothing,
    },
    stft::stft,
    window::windowing,
};

pub mod dct;
pub mod erb;
pub mod filter;
pub mod framing;
pub mod gammatone;
pub mod mean_power_normalization;
pub mod medium_time_processing;
pub mod normalizer;
pub mod stft;
pub mod window;

pub static PNCC_NUM_CEPS: usize = 13;

pub struct PNCCArgs<'s> {
    signal: &'s [f32],
    samples_rate: f32,
    nfft: usize,
    power: u32,
    n_filts: u8,
    num_ceps: u8,
}

impl<'s> PNCCArgs<'s> {
    pub fn new(signal: &'s [f32]) -> Self {
        Self {
            signal,
            n_filts: 40,
            nfft: 1024,
            num_ceps: 13,
            power: 2,
            samples_rate: 16000.,
        }
    }

    pub fn get_signal(&self) -> &'s [f32] {
        self.signal
    }

    pub fn get_n_filts(&self) -> u8 {
        self.n_filts
    }

    pub fn get_nfft(&self) -> usize {
        self.nfft
    }

    pub fn get_num_ceps(&self) -> u8 {
        self.num_ceps
    }

    pub fn get_power(&self) -> u32 {
        self.power
    }

    pub fn get_samples_rate(&self) -> f32 {
        self.samples_rate
    }

    pub fn get_win_hop() -> f32 {
        0.010
    }

    pub fn get_win_len() -> f32 {
        0.025
    }
}

/// Calcule les coefficients delta (dérivées temporelles) d'une matrice de cepstre.
///
/// # Arguments
/// * `ceps`  - Matrice (n_frames × n_coeffs)
/// * `win`   - Largeur de la fenêtre ; `h = win / 2` trames de chaque côté
///
/// # Retourne
/// Matrice (n_frames × n_coeffs) des delta-coefficients
pub fn deltas(ceps: &Array2<f32>, win: usize) -> Array2<f32> {
    let (n_frames, n_coeffs) = ceps.dim();

    // Garde-fous
    let h = win / 2;
    if n_frames == 0 || n_coeffs == 0 || h == 0 {
        return Array2::zeros((n_frames, n_coeffs));
    }

    // Norme : 2 * Σ k² pour k = 1..=h  (pondération quadratique standard)
    let norm: f32 = 2.0 * (1..=h).map(|k| (k * k) as f32).sum::<f32>();

    let mut delta = Array2::zeros((n_frames, n_coeffs));

    // Boucle sur les décalages k d'abord → meilleure localité mémoire
    for k in 1..=h {
        let kf = k as f32;
        for f in 0..n_frames {
            // Réplication bord (edge padding) : clamping d'indice
            let prev = f.saturating_sub(k);
            let next = (f + k).min(n_frames - 1);

            // Différence vectorisée sur tous les coefficients de la trame
            // delta[f] += k * (ceps[f+k] - ceps[f-k])
            let diff = &ceps.row(next) - &ceps.row(prev);
            delta.row_mut(f).scaled_add(kf, &diff);
        }
    }

    // Division par la norme en une seule passe (évite n_frames * n_coeffs divisions)
    delta /= norm;
    delta
}

pub fn pncc(args: &PNCCArgs) -> Array2<f32> {
    let weight_n = 4;
    let asymetric_noise_suppression_threshold = 0.0;

    // Pre-emphasis
    let pre_emphasis = filter(&[1., -0.97], &[1.], args.get_signal());

    // framing
    let framed = framing(
        &pre_emphasis,
        args.get_samples_rate(),
        PNCCArgs::get_win_len(),
        PNCCArgs::get_win_hop(),
    );

    // windowing
    let windowed = windowing(
        &framed.0,
        framed.1.try_into().unwrap(),
        &window::WindowType::Hamming,
    );

    // Compute STFT
    let stft_ed = stft(&windowed, args.get_nfft());

    // Apply power Spectrum
    let stft_powerd = stft_ed
        .map(|x| (1.0 / args.get_nfft() as f32) * x.powi(args.get_power().try_into().unwrap()));

    let filter_bank = fft2gammatonemx(
        args.get_samples_rate(),
        200.0,
        args.get_n_filts(),
        args.get_nfft(),
        4,
    );

    let gm_freq_integ = stft_powerd.dot(&filter_bank.t());

    // Medium time processing
    // 1. medium time power caculations
    let medium_time_powered =
        medium_time_power_calculation(&gm_freq_integ, args.get_power().try_into().unwrap());

    // 2. asymmetric noise suppression with temporal masking
    // 2.1. asymmetric low pass filtering
    let lower_envelope = asymmetric_lowpass_filtering(&medium_time_powered, 0.999, 0.5);

    // 2.2. Subtract filtering output from the input
    let subtracted_lower_envelope = medium_time_powered.clone() - lower_envelope.clone();

    // 2.3. half wave rectification
    let half_wave_rectified_signal = subtracted_lower_envelope.map(|x| {
        if x < &asymetric_noise_suppression_threshold {
            0.0
        } else {
            *x
        }
    });

    // 2.4 floor level: lower envelope of the rectifier output/ rectified signal
    let floor_level = asymmetric_lowpass_filtering(&half_wave_rectified_signal, 0.999, 0.5);

    // 2.5. temporal masking
    let temporal_masked_signal = temporal_masking(&half_wave_rectified_signal, 0.85, 0.2);

    // 2.6. switch excitation or non-excitation
    let final_output = switch_excitation_or_non_excitation(
        &temporal_masked_signal,
        &floor_level,
        &lower_envelope,
        &medium_time_powered,
        2.0,
    );

    // 3. weight smoothing
    let spectral_weight_smoothing = weight_smoothing(
        &final_output,
        &medium_time_powered,
        args.get_n_filts().into(),
        weight_n,
    );

    // time frequency normalization
    let transfer_function = {
        let (rows, cols) = gm_freq_integ.dim();
        let mut array = Array2::<f32>::zeros([rows, cols]);
        for row in 0..rows {
            for col in 0..cols {
                array[[row, col]] =
                    gm_freq_integ[[row, col]] * spectral_weight_smoothing[[row, col]];
            }
        }
        array
    };

    // mean power normalization
    let normalized_power =
        mean_power_normalization(&transfer_function, 0.999, args.get_n_filts(), 1.0);

    // power law non-linearity
    let power_law_nonlinearity = normalized_power.map(|x| x.powf(1.0 / 15.0));

    let num_ceps: usize = args.get_num_ceps().into();
    // dct

    dct_2d_axis(&power_law_nonlinearity, dct::Norm::Ortho, dct::Axis::Rows)
        .slice(s![.., ..num_ceps])
        .into_owned()
    // mean_variance_normalizer(&pnccs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_preserved() {
        let ceps = Array2::<f32>::zeros((10, 13));
        let d = deltas(&ceps, 5);
        assert_eq!(d.dim(), (10, 13));
    }

    #[test]
    fn test_zero_input_gives_zero_delta() {
        let ceps = Array2::<f32>::zeros((5, 4));
        let d = deltas(&ceps, 3);
        assert!(d.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_edge_cases() {
        let ceps = Array2::<f32>::zeros((3, 2));
        // h=0 → retourne zéros
        let d = deltas(&ceps, 1);
        assert_eq!(d.dim(), (3, 2));
        assert!(d.iter().all(|&v| v == 0.0));

        // Matrice vide
        let empty = Array2::<f32>::zeros((0, 13));
        let d = deltas(&empty, 5);
        assert_eq!(d.dim(), (0, 13));
    }

    #[test]
    fn test_linear_ramp_has_constant_delta() {
        // Si ceps[f][j] = f, alors delta[f][j] doit être constant ≈ 1
        let n = 20usize;
        let ceps = Array2::from_shape_fn((n, 1), |(f, _)| f as f32);
        let d = deltas(&ceps, 5);
        // Trames intérieures (loin des bords) doivent donner delta ≈ 1.0
        for f in 3..n - 3 {
            let diff = (d[(f, 0)] - 1.0).abs();
            assert!(diff < 1e-10, "frame {f}: delta = {}", d[(f, 0)]);
        }
    }
}
