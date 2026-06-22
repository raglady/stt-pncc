use ndarray::Array2;
/// Apply mean power normalization according to Kim et al.
///
/// # Arguments
///
/// * `t` - Transfer function matrix (rows × columns)
/// * `lam_myu` - Time constant (λ_μ)
/// * `n_filts` - Total number of channels/filters
/// * `k` - Arbitrary constant
///
/// # Returns
///
/// Normalized mean power matrix
///
/// # Algorithm
///
/// The normalization is computed as:
///
/// ```text
/// μ[m] = λ_μ · μ[m-1] + (1 - λ_μ) / L · Σ(l=0 to L-1) T[m, l]
///
/// U[m, l] = k · T[m, l] / μ[m]
/// ```
///
/// where λ_μ is the time constant and L is the number of filters.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use pncc::mean_power_normalization::mean_power_normalization;
///
/// let t = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let result = mean_power_normalization(&t, 0.9, 3, 1.0);
///
pub fn mean_power_normalization(
    transfer_function: &Array2<f64>,
    lam_myu: f64,
    n_filts: u8,
    k: f64,
) -> Array2<f64> {
    let (rows, cols) = transfer_function.dim();
    let mut myu = vec![0.0; rows];
    myu[0] = 0.0001;

    for m in 1..rows {
        let mut sum = 0.0;
        for s in 0..n_filts {
            sum += transfer_function[[m, s.into()]];
        }
        myu[m] = lam_myu * myu[m - 1] + (1.0 - lam_myu) / n_filts as f64 * sum;
    }

    let mut normalized_power = Array2::<f64>::zeros([rows, cols]);
    normalized_power
        .indexed_iter_mut()
        .for_each(|((i, j), val)| *val = k * transfer_function[[i, j]] / myu[i]);
    normalized_power
}
