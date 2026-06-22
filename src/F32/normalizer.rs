use ndarray::{Array2, ArrayView2, Axis};

pub fn mean_variance_normalizer(input: &Array2<f32>) -> Array2<f32> {
    // Calculate mean for each column
    let mean_col = input.mean_axis(Axis(0)).unwrap();
    let std = input.std_axis(Axis(0), 0.);
    &(input - mean_col) / &std
}

pub fn variance_normalizer(input: &Array2<f32>) -> Array2<f32> {
    let std = input.std_axis(Axis(0), 0.);
    input / std
}

pub fn translate_to_positive(input: ArrayView2<f32>) -> Array2<f32> {
    let min = input.iter().min_by(|a, b| a.total_cmp(b)).unwrap();

    let mut output = input.to_owned();
    output.par_mapv_inplace(|v| v + min.abs());
    output
}
