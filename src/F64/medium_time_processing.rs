use ndarray::{Array2, aview1};

///
///      Compute the medium time power calulations according to [Kim]_ .
///
///    Args:
///        power_stft_signal (Array2D) : signal stft power.
///        m           (int) : the temporal integration factor.
///
///    Returns:
///        (Array2D) : medium time power values.
///
pub fn medium_time_power_calculation(power_stft_signal: &Array2<f64>, m: usize) -> Array2<f64> {
    let (rows, cols) = power_stft_signal.dim();
    let mut medium_time_power = Array2::<f64>::zeros([rows, cols]);

    // Create padded array
    let padded_rows = rows + 2 * m;
    let mut padded = Array2::<f64>::zeros([padded_rows, cols]);

    // Copy data to padded array
    for i in 0..rows {
        for j in 0..cols {
            padded[[i + m, j]] = power_stft_signal[[i, j]];
        }
    }

    // Calculate medium time power
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..(2 * m + 1) {
                sum += padded[[i + k, j]];
            }
            medium_time_power[[i, j]] = sum / (2 * m + 1) as f64;
        }
    }

    medium_time_power
}

///
///     Apply asymmetric lowpass filter according to [Kim]_ .
///
///    Args:
///        rectified_signal (Array2D) : rectified signal.
///        lm_a               (float) : filter parameter; lambda a. Default: 0.999
///        lm_b               (float) : filter parameter; lambda b. Default: 0.5
///
///    Returns:
///        (Array2D) : filtered signal.
///
pub fn asymmetric_lowpass_filtering(
    rectified_signal: &Array2<f64>,
    lm_a: f64,
    lm_b: f64,
) -> Array2<f64> {
    let (rows, cols) = rectified_signal.dim();
    let mut floor_level = Array2::<f64>::zeros([rows, cols]);

    // Initialize first row
    floor_level.row_mut(0).assign(&aview1(
        &rectified_signal
            .row(0)
            .iter()
            .map(|x| x * 0.9)
            .collect::<Vec<f64>>(),
    ));

    // Process remaining rows
    for m in 0..rows {
        let a = if m == 0 { rows - 1 } else { m - 1 };
        for i in 0..cols {
            let value = if rectified_signal[[m, i]] >= floor_level[[a, i]] {
                lm_a * floor_level[[a, i]] + (1.0 - lm_a) * rectified_signal[[m, i]]
            } else {
                lm_b * floor_level[[a, i]] + (1. - lm_b) * rectified_signal[[m, i]]
            };
            floor_level[[m, i]] = value;
        }
    }
    floor_level
}

///
///     Args:
///        rectified_signal (Array2D) : rectified signal.
///        lam_t             (float) : the forgetting factor-
///        myu_t             (float) : the recognition accuracy.
///
///    Returns:
///        (Array2D) : temporal_masked_signal = temporal_masking(rectified_signal)
pub fn temporal_masking(rectified_signal: &Array2<f64>, lam_t: f64, myu_t: f64) -> Array2<f64> {
    let (rows, cols) = rectified_signal.dim();
    let mut temporal_masked_signal = Array2::<f64>::zeros([rows, cols]);
    let mut online_peak_power = Array2::<f64>::zeros([rows, cols]);

    // Initialize first row
    for j in 0..cols {
        temporal_masked_signal[[0, j]] = rectified_signal[[0, j]];
        online_peak_power[[0, j]] = rectified_signal[[0, j]];
    }

    // Process remaining rows
    for m in 1..rows {
        for l in 0..cols {
            let peak = f64::max(
                lam_t * online_peak_power[[m - 1, l]],
                rectified_signal[[m, l]],
            );
            online_peak_power[[m, l]] = peak;

            let masked = if rectified_signal[[m, l]] >= lam_t * online_peak_power[[m - 1, l]] {
                rectified_signal[[m, l]]
            } else {
                myu_t * online_peak_power[[m - 1, l]]
            };
            temporal_masked_signal[[m, l]] = masked;
        }
    }

    temporal_masked_signal
}

pub fn switch_excitation_or_non_excitation(
    temporal_masked_signal: &Array2<f64>,
    floor_level: &Array2<f64>,
    lower_envelope: &Array2<f64>,
    medium_time_power: &Array2<f64>,
    c: f64,
) -> Array2<f64> {
    let (rows, cols) = temporal_masked_signal.dim();
    let mut result = Array2::<f64>::zeros([rows, cols]);

    result.indexed_iter_mut().for_each(|((i, j), val)| {
        *val = if medium_time_power[[i, j]] >= c * lower_envelope[[i, j]] {
            temporal_masked_signal[[i, j]]
        } else {
            floor_level[[i, j]]
        };
    });
    result
}

///    Apply spectral weight smoothing according to [Kim]_.
///
///    Args:
///        final_output (Array2D) :
///        medium_time_power (Array2D) : medium time power
///        n_filts            (int) : total number of channels / filters
///        N                 (int) :
///
///    Returns:
///        (Array2D) : time-averaged frequency-averaged transfer function.
///
pub fn weight_smoothing(
    final_output: &Array2<f64>,
    medium_time_power: &Array2<f64>,
    n_filts: usize,
    n: u8,
) -> Array2<f64> {
    let (rows, cols) = final_output.dim();
    let mut spectral_weight_smoothing = Array2::<f64>::zeros([rows, cols]);

    spectral_weight_smoothing
        .indexed_iter_mut()
        .for_each(|((m, l_idx), val)| {
            let l_1 = l_idx.saturating_sub(n as usize);
            let l_2 = (l_idx + n as usize).min(n_filts);

            let mut sum = 0.0;
            for l_ in l_1..l_2 {
                // println!("med: {}", medium_time_power.get(m, l_));
                sum += final_output[[m, l_]] / medium_time_power[[m, l_]].max(f64::EPSILON);
            }

            *val = sum / (l_2 as isize - l_1 as isize + 1) as f64;
        });

    spectral_weight_smoothing
}
