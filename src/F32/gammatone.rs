use std::f32::consts::PI;

use ndarray::{Array2, aview1};
use num_complex::{Complex32, ComplexFloat};

use super::erb::{EAR_Q, MIN_BW, erb_space};

pub fn fft2gammatonemx(
    samples_rate: f32,
    low_freq: f32,
    n_filts: u8,
    nfft: usize,
    order: u8,
) -> Array2<f32> {
    let mut cfreqs = erb_space(low_freq, samples_rate / 2.0, n_filts);
    cfreqs.reverse();

    let _width = 1;

    let g_tord = 4;

    let ucirc: Vec<Complex32> = (0..=nfft / 2)
        .map(|i| Complex32::new(0., 2.0 * PI * i as f32 / nfft as f32).exp())
        .collect();

    let justpoles = 0;
    let mut gain = vec![0.; n_filts.into()];
    let mut wts = Array2::<f32>::zeros([n_filts.into(), nfft / 2 + 1]);

    for i in 0..n_filts as usize {
        let cf = cfreqs[i];
        let b_tmp = 1.019
            * 2.0
            * PI
            * (((cf / EAR_Q).powi(order.into()) + MIN_BW.powi(order.into()))
                .powf(1.0 / order as f32));
        let r = (-b_tmp / samples_rate).exp();
        let theta = 2.0 * PI * cf / samples_rate;
        let pole = r * Complex32::new(0., theta).exp();

        if justpoles == 1 {
            // point on unit circle of maximum gain, from differentiating magnitude
            let cosomegamax = (1.0 + r * r) / (2.0 * r) * (theta).cos();
            let omegamax = if (cosomegamax).abs() > 1. {
                if theta < PI / 2.0 { 0.0 } else { PI }
            } else {
                (cosomegamax).acos()
            };

            let center = Complex32::new(0., omegamax).exp();
            let g = ((pole - center) * (pole.conj() - center))
                .abs()
                .powi(g_tord);
            wts.row_mut(i).assign(&aview1(
                &ucirc
                    .iter()
                    .map(|v: &num_complex::Complex<f32>| {
                        g * (((pole - v) * (pole.conj() - v)).abs().powi(-g_tord))
                    })
                    .collect::<Vec<f32>>(),
            ));
        } else {
            // poles and zeros, following Malcolm's MakeERBFilter
            let t = 1.0 / samples_rate;
            let a11 = -(2.0 * t * (2.0 * cf * PI * t).cos() / (b_tmp * t).exp()
                + 2.0 * (3.0 + (2.0f32).powf(1.5)).sqrt() * t * (2.0 * cf * PI * t).sin()
                    / (b_tmp * t).exp())
                / 2.0;
            let a12 = -(2.0 * t * (2.0 * cf * PI * t).cos() / (b_tmp * t).exp()
                - 2.0 * (3.0 + (2.0f32).powf(1.5)).sqrt() * t * (2.0 * cf * PI * t).sin()
                    / (b_tmp * t).exp())
                / 2.0;
            let a13 = -(2.0 * t * (2.0 * cf * PI * t).cos() / (b_tmp * t).exp()
                + 2.0 * (3.0 - (2.0f32).powf(1.5)).sqrt() * t * (2.0 * cf * PI * t).sin()
                    / (b_tmp * t).exp())
                / 2.0;
            let a14 = -(2.0 * t * (2.0 * cf * PI * t).cos() / (b_tmp * t).exp()
                - 2.0 * (3.0 - (2.0f32).powf(1.5)).sqrt() * t * (2.0 * cf * PI * t).sin()
                    / (b_tmp * t).exp())
                / 2.0;

            let zros: Vec<f32> = [a11, a12, a13, a14].iter().map(|x| -x / t).collect();

            gain[i] = ((-2.0 * Complex32::new(0., 4.0 * cf * PI * t).exp() * t
                + 2.0
                    * Complex32::new(-(b_tmp * t), 2.0 * cf * PI * t).exp()
                    * t
                    * ((2.0 * cf * PI * t).cos()
                        - (3.0 - (2.0f32).powf(1.5)).sqrt() * (2.0 * cf * PI * t).sin()))
                * (-2.0 * Complex32::new(0., 4.0 * cf * PI * t).exp() * t
                    + 2.0
                        * Complex32::new(-(b_tmp * t), 2.0 * cf * PI * t).exp()
                        * t
                        * ((2.0 * cf * PI * t).cos()
                            + (3.0 - (2.0f32).powf(1.5)).sqrt() * (2.0 * cf * PI * t).sin()))
                * (-2.0 * Complex32::new(0., 4.0 * cf * PI * t).exp() * t
                    + 2.0
                        * Complex32::new(-(b_tmp * t), 2.0 * cf * PI * t).exp()
                        * t
                        * ((2.0 * cf * PI * t).cos()
                            - (3.0 + (2.0f32).powf(1.5)).sqrt() * (2.0 * cf * PI * t).sin()))
                * (-2.0 * Complex32::new(0., 4.0 * cf * PI * t).exp() * t
                    + 2.0
                        * Complex32::new(-(b_tmp * t), 2.0 * cf * PI * t).exp()
                        * t
                        * ((2.0 * cf * PI * t).cos()
                            + (3.0 + (2.0f32).powf(1.5)).sqrt() * (2.0 * cf * PI * t).sin()))
                / (-2.0 / (2.0 * b_tmp * t).exp()
                    - 2.0 * Complex32::new(0., 4.0 * cf * PI * t).exp()
                    + 2.0 * (1.0 + Complex32::new(0., 4.0 * cf * PI * t).exp())
                        / (b_tmp * t).exp())
                .powf(4.0))
            .abs();
            wts.row_mut(i).assign(&aview1(
                &ucirc
                    .iter()
                    .map(|v: &num_complex::Complex<f32>| {
                        ((t.powf(4.0)) / gain[i])
                            * (v - zros[0]).abs()
                            * (v - zros[1]).abs()
                            * (v - zros[2]).abs()
                            * (v - zros[3]).abs()
                            * (((pole - v) * (pole.conj() - v)).abs().powi(-g_tord))
                    })
                    .collect::<Vec<f32>>(),
            ));
        }

        // make sure all filters has max value = 1.0
        let value = {
            let row = wts.row(i).to_vec();
            let mut max = 0.;
            for elem in row.iter() {
                if elem > &max {
                    max = *elem;
                }
            }
            row.iter().map(|c| c / max).collect::<Vec<f32>>()
        };
        wts.row_mut(i).assign(&aview1(&value));
    }
    wts
}
