use std::f32::consts::PI;

use ndarray::Array2;
use ndarray::aview1;
use num_complex::Complex32;
use num_complex::ComplexFloat;

use super::filter::filter;

// ERB Constant
pub const EAR_Q: f32 = 9.26449;
pub const MIN_BW: f32 = 24.7;

/// Computes an array of N frequencies uniformly spaced between
/// highFreq and lowFreq on an ERB scale.
///
/// For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
/// "Suggested formulae for calculating auditory-filter bandwidths and
/// excitation patterns," J. Acoust. Soc. Am. 74, 750-753.
///
/// # Arguments
///
/// * `low_freq` - Lower frequency bound (default: 100.0)
/// * `high_freq` - Upper frequency bound (default: 11025.0)
/// * `n` - Number of frequency points (or number of filters)
///
/// # Returns
///
/// Vector of frequencies spaced on an ERB scale( vector of center frequencies )
pub fn erb_space(low_freq: f32, high_freq: f32, n: u8) -> Vec<f32> {
    // All of the following expressions are derived in Apple TR #35, "An
    // Efficient Implementation of the Patterson-Holdsworth Cochlear
    // Filter Bank." See pages 33-34.
    let vec: Vec<f32> = (1..=n)
        .map(|i| {
            let exponent = ((i as f32)
                * ((low_freq + (EAR_Q * MIN_BW)) / (high_freq + (EAR_Q * MIN_BW))).ln()
                / (n as f32))
                .exp();
            -(EAR_Q * MIN_BW) + exponent * (high_freq + (EAR_Q * MIN_BW))
        })
        .collect();
    //  vec.reverse();
    vec
}

pub fn erb_filter_bank(
    input_puls: &[f32],
    samples_rate: f32,
    low_freq: f32,
    n_filts: u8,
    order: u8,
) -> Array2<f32> {
    assert!(!input_puls.is_empty());
    assert!(samples_rate > low_freq);
    assert!(order > 0);
    assert!(low_freq >= 0.);
    assert!(n_filts > 0);

    let t = 1.0 / samples_rate;
    let cf = erb_space(low_freq, samples_rate / 2.0, n_filts); // centered frequencies

    let mut b = vec![0.; n_filts.into()];
    let mut gain = b.clone();
    let mut feedback = Array2::<f32>::zeros([n_filts.into(), 9]);
    let mut forward = Array2::<f32>::zeros([n_filts.into(), 5]);
    let mut output = Array2::<f32>::zeros([n_filts.into(), input_puls.len()]);

    for i in 0..n_filts as usize {
        let temp_b = 1.019
            * 2.0
            * PI
            * (((cf[i] / EAR_Q).powi(order.into()) + MIN_BW.powi(order.into()))
                .powf(1.0 / order as f32));
        b[i] = temp_b;

        let gain_tmp = ((-2.0 * (Complex32::new(0., 4.0 * cf[i] * PI * t)).exp() * t
            + 2.0
                * Complex32::new(-(temp_b * t), 2.0 * cf[i] * PI * t).exp()
                * t
                * ((2.0 * cf[i] * PI * t).cos()
                    - (3.0 - (2.0f32).powf(1.5)).sqrt() * (2.0 * cf[i] * PI * t).sin()))
            * (-2.0 * Complex32::new(0., 4.0 * cf[i] * PI * t).exp() * t
                + 2.0
                    * Complex32::new(-(temp_b * t), 2.0 * cf[i] * PI * t).exp()
                    * t
                    * ((2.0 * cf[i] * PI * t).cos()
                        + (3.0 - (2.0f32).powf(1.5)).sqrt() * (2.0 * cf[i] * PI * t).sin()))
            * (-2.0 * Complex32::new(0., 4.0 * cf[i] * PI * t).exp() * t
                + 2.0
                    * Complex32::new(-(temp_b * t), 2.0 * cf[i] * PI * t).exp()
                    * t
                    * ((2.0 * cf[i] * PI * t).cos()
                        - (3.0 + (2.0f32).powf(1.5)).sqrt() * (2.0 * cf[i] * PI * t).sin()))
            * (-2.0 * Complex32::new(0., 4.0 * cf[i] * PI * t).exp() * t
                + 2.0
                    * Complex32::new(-(temp_b * t), 2.0 * cf[i] * PI * t).exp()
                    * t
                    * ((2.0 * cf[i] * PI * t).cos()
                        + (3.0 + (2.0f32).powf(1.5)).sqrt() * (2.0 * cf[i] * PI * t).sin()))
            / (-2.0 / (2.0 * temp_b * t).exp()
                - 2.0 * Complex32::new(0., 4.0 * cf[i] * PI * t).exp()
                + 2.0 * (1.0 + Complex32::new(0., 4.0 * cf[i] * PI * t).exp())
                    / (temp_b * t).exp())
            .powf(4.0))
        .abs();
        gain[i] = gain_tmp;

        forward[[i, 0]] = t.powf(4.0) / gain_tmp;
        forward[[i, 1]] =
            -4.0 * t.powf(4.0) * (2.0 * cf[i] * PI * t).cos() / (temp_b * t).exp() / gain_tmp;
        forward[[i, 2]] =
            6.0 * t.powf(4.0) * (4.0 * cf[i] * PI * t).cos() / (2.0 * temp_b * t).exp() / gain_tmp;
        forward[[i, 3]] =
            -4.0 * t.powf(4.0) * (6.0 * cf[i] * PI * t).cos() / (3.0 * temp_b * t).exp() / gain_tmp;
        forward[[i, 4]] =
            t.powf(4.0) * (8.0 * cf[i] * PI * t).cos() / (4.0 * temp_b * t).exp() / gain_tmp;
        feedback[[i, 0]] = 1.;
        feedback[[i, 1]] = -8.0 * (2.0 * cf[i] * PI * t).cos() / (temp_b * t).exp();
        feedback[[i, 2]] =
            4.0 * (4.0 + 3.0 * (4.0 * cf[i] * PI * t).cos()) / (2.0 * temp_b * t).exp();
        feedback[[i, 3]] = -8.0
            * (6.0 * (2.0 * cf[i] * PI * t).cos() + (6.0 * cf[i] * PI * t).cos())
            / (3.0 * temp_b * t).exp();
        feedback[[i, 4]] = 2.0
            * (18.0 + 16.0 * (4.0 * cf[i] * PI * t).cos() + (8.0 * cf[i] * PI * t).cos())
            / (4.0 * temp_b * t).exp();
        feedback[[i, 5]] = 8.0
            * (6.0 * (2.0 * cf[i] * PI * t).cos() + (6.0 * cf[i] * PI * t).cos())
            / (5.0 * temp_b * t).exp();
        feedback[[i, 6]] =
            4.0 * (4.0 + 3.0 * (4.0 * cf[i] * PI * t).cos()) / (6.0 * temp_b * t).exp();
        feedback[[i, 7]] = -8.0 * (2.0 * cf[i] * PI * t).cos() / (7.0 * temp_b * t).exp();
        feedback[[i, 8]] = (-8.0 * temp_b * t).exp();

        output.row_mut(i).assign(&aview1(&filter(
            forward.row(i).as_slice().unwrap(),
            feedback.row(i).as_slice().unwrap(),
            input_puls,
        )));
    }
    output
}

// Example usage:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erb_space_custom() {
        let cf_array = erb_space(100.0, 8000.0, 50);
        println!("cf_array: {:?}", cf_array);
        assert_eq!(cf_array.len(), 50);
        // First value should be close to lowFreq
        assert!(cf_array[49] > 99.9 && cf_array[49] < 200.0);
        // Last value should be close to highFreq
        assert!(cf_array[0] > 7000.0 && cf_array[0] < 8000.0);
    }
}
