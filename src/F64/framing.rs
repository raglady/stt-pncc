use ndarray::Array2;

/// return frames (Frames array, frame length)
pub fn framing(
    signal: &[f64],
    samples_rate: f64,
    win_len: f64,
    win_hop: f64,
) -> (Array2<f64>, u32) {
    let frame_len: f64 = samples_rate * win_len;
    if frame_len > u32::MAX as f64 {
        panic!("frame_len is greater than u32 Max");
    }
    let frame_len = frame_len.ceil() as u32;

    let frame_hop = samples_rate * win_hop;

    if frame_hop > u32::MAX as f64 {
        panic!("frame_hop is greater than u32 Max");
    }

    let frame_hop = frame_hop.ceil() as u32;

    let mut n_frames =
        (signal.len() as f64 - frame_len as f64 + frame_hop as f64) / frame_hop as f64;

    if n_frames < 0. {
        n_frames = 1.;
    }

    if n_frames > u32::MAX as f64 {
        panic!("n_frames is greater than u32 Max");
    }

    let n_frames = n_frames.ceil() as u32;

    let mut frames = Array2::<f64>::zeros([n_frames as usize, frame_len as usize]);

    let mut start = 0usize;
    let mut row = 0;
    while row < n_frames {
        let mut i = start;
        let mut col = 0;
        let end = start + frame_len as usize;
        while i < end && i < signal.len() {
            frames[[row.try_into().unwrap(), col]] = signal[i];
            i += 1;
            col += 1;
        }
        row += 1;
        start += frame_hop as usize;
    }
    (frames, frame_len)
}
