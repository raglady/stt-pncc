use ndarray::Array2;
use rustfft::{FftPlanner, num_complex::Complex};

pub fn stft(frames: &Array2<f32>, nfft: usize) -> Array2<f32> {
    let (rows, cols) = frames.dim();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);

    let n_freq = nfft / 2 + 1;
    let mut new_frames = Array2::<f32>::zeros([rows, n_freq]);

    for i in 0..rows {
        let mut buffer = vec![Complex::new(0.0, 0.0); nfft];
        for j in 0..cols {
            buffer[j] = Complex::new(frames[[i, j]], 0.);
        }
        fft.process(&mut buffer);
        for j in 0..n_freq {
            let magnitude = buffer[j].norm();
            new_frames[[i, j]] = magnitude;
        }
    }
    new_frames
}
