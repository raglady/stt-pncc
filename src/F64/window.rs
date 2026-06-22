use std::f64::consts::PI;

use ndarray::Array2;

pub enum WindowType {
    None,
    Hann,
    Hamming,
    Blackman,
}

pub fn windowing(frames: &Array2<f64>, _frame_len: usize, window_type: &WindowType) -> Array2<f64> {
    let (n_rows, n_cols) = frames.dim();

    let window = match window_type {
        WindowType::None => vec![0.0; n_cols],
        WindowType::Hann => {
            let mut w = vec![0.0; n_cols];
            for (i, item) in w.iter_mut().enumerate().take(n_cols) {
                *item = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n_cols - 1) as f64).cos());
            }
            w
        }
        WindowType::Hamming => {
            let mut w = vec![0.0; n_cols];
            for (i, item) in w.iter_mut().enumerate().take(n_cols) {
                *item = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n_cols - 1) as f64).cos();
            }
            w
        }
        WindowType::Blackman => {
            let mut w = vec![0.0; n_cols];
            for (i, item) in w.iter_mut().enumerate().take(n_cols) {
                let x = 2.0 * PI * i as f64 / (n_cols - 1) as f64;
                *item = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
            }
            w
        }
    };
    let mut new_frames = Array2::<f64>::zeros([n_rows, n_cols]);

    for i in 0..n_rows {
        for j in 0..n_cols {
            new_frames[[i, j]] = frames[[i, j]] * window[j]
        }
    }
    new_frames
}
