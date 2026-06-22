use std::f64::consts::PI;

use ndarray::Array2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    /// No normalization.
    ///
    /// ⚠️  This follows the convention:
    ///     X[k] = 2 · Σ x[n] · cos(π·k·(2n+1) / 2N)
    ///
    /// This matches scipy's `norm=None` / `norm="backward"` output exactly.
    /// Earlier versions of this crate omitted the leading factor of 2 — if
    /// you were compensating for that manually, remove that compensation now.
    None,

    /// Orthonormal normalisation (matches scipy `norm="ortho"`).
    ///
    /// Forward scale factors:
    ///   k = 0  →  √(1/N)
    ///   k > 0  →  √(2/N)
    Ortho,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Apply the 1-D transform along each row independently.
    Rows,
    /// Apply the 1-D transform along each column independently.
    Columns,
    /// Full separable 2-D transform (rows then columns for forward,
    /// columns then rows for inverse — identical result, just conventional).
    Both,
}

// ---------------------------------------------------------------------------
// Cosine look-up table
// ---------------------------------------------------------------------------

/// Precomputes the N×N table  cos(π · k · (2n+1) / 2N)  for k,n ∈ [0, N).
///
/// Indexed as `table[k * n_size + n]`.
/// Avoids recomputing identical cosine values on every inner loop iteration.
fn build_cos_table(n: usize) -> Vec<f64> {
    let mut table = vec![0.0_f64; n * n];
    let denom = 2.0 * n as f64;
    for k in 0..n {
        for j in 0..n {
            table[k * n + j] = ((2 * j + 1) as f64 * k as f64 * PI / denom).cos();
        }
    }
    table
}

// ---------------------------------------------------------------------------
// Forward DCT-II
// ---------------------------------------------------------------------------

/// Applies the 1-D DCT-II independently to every row.
///
/// Memory access pattern: row-major, fully sequential → cache-friendly.
fn dct_along_rows(input: &Array2<f64>, norm: Norm) -> Array2<f64> {
    let (rows, cols) = input.dim();
    let cos = build_cos_table(cols);
    let mut output = Array2::<f64>::zeros((rows, cols));

    for i in 0..rows {
        for v in 0..cols {
            let mut sum = 0.0;
            for j in 0..cols {
                sum += input[[i, j]] * cos[v * cols + j];
            }

            // FIX #1: Norm::None now includes the leading factor of 2,
            // matching scipy's unnormalised DCT-II convention.
            output[[i, v]] = match norm {
                Norm::Ortho => {
                    if v == 0 {
                        (1.0 / cols as f64).sqrt() * sum
                    } else {
                        (2.0 / cols as f64).sqrt() * sum
                    }
                }
                Norm::None => 2.0 * sum,
            };
        }
    }
    output
}

/// Applies the 1-D DCT-II independently to every column.
///
/// FIX #6: Instead of iterating with `i` in the inner loop (strided / cache-hostile),
/// we transpose the matrix, reuse the cache-friendly row kernel, then transpose back.
fn dct_along_columns(input: &Array2<f64>, norm: Norm) -> Array2<f64> {
    // t() returns a *view* (no copy); to_owned() materialises it into a
    // contiguous row-major Array2 so that dct_along_rows gets sequential access.
    let transposed = input.t().to_owned();
    let result = dct_along_rows(&transposed, norm);
    result.t().to_owned()
}

/// Performs a separable 2-D DCT-II with the given normalisation and axis mode.
pub fn dct_2d_axis(input: &Array2<f64>, norm: Norm, axis: Axis) -> Array2<f64> {
    match axis {
        Axis::Rows => dct_along_rows(input, norm),
        Axis::Columns => dct_along_columns(input, norm),
        // Separability: applying rows then columns (or vice-versa) gives the
        // same result.  We use rows→columns here to match scipy's 2-D DCT order.
        Axis::Both => {
            let temp = dct_along_rows(input, norm);
            dct_along_columns(&temp, norm)
        }
    }
}

/// Convenience wrapper: full 2-D DCT-II with orthonormal normalisation.
///
/// Equivalent to `dct_2d_axis(input, Norm::Ortho, Axis::Both)`.
pub fn dct_2d(input: &Array2<f64>) -> Array2<f64> {
    dct_2d_axis(input, Norm::Ortho, Axis::Both)
}

// ---------------------------------------------------------------------------
// Inverse DCT-II  (= DCT-III, scaled)
// ---------------------------------------------------------------------------

/// Applies the inverse 1-D DCT-II to every row.
fn idct_along_rows(input: &Array2<f64>, norm: Norm) -> Array2<f64> {
    let (rows, cols) = input.dim();
    let cos = build_cos_table(cols);
    let mut output = Array2::<f64>::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for v in 0..cols {
                let scale = match norm {
                    Norm::Ortho => {
                        if v == 0 {
                            (1.0 / cols as f64).sqrt()
                        } else {
                            (2.0 / cols as f64).sqrt()
                        }
                    }
                    // For the scipy-compatible Norm::None forward  Y[k] = 2·Σ x·cos(…)
                    // the correct inverse is:
                    //   x[j] = (1/(2N)) · (Y[0] + 2·Σ_{k>0} Y[k]·cos(…))
                    // i.e. scale = [1, 2, 2, …]  then divide by 2N.
                    Norm::None => {
                        if v == 0 {
                            1.0
                        } else {
                            2.0
                        }
                    }
                };
                sum += scale * input[[i, v]] * cos[v * cols + j];
            }
            output[[i, j]] = match norm {
                Norm::Ortho => sum,
                Norm::None => sum / (2.0 * cols as f64), // ← divide by 2N, not N
            };
        }
    }
    output
}

/// Applies the inverse 1-D DCT-II to every column.
///
/// FIX #6: same transpose trick as the forward column transform.
fn idct_along_columns(input: &Array2<f64>, norm: Norm) -> Array2<f64> {
    let transposed = input.t().to_owned();
    let result = idct_along_rows(&transposed, norm);
    result.t().to_owned()
}

/// Performs a separable 2-D inverse DCT-II with the given normalisation and axis mode.
pub fn idct_2d_axis(input: &Array2<f64>, norm: Norm, axis: Axis) -> Array2<f64> {
    match axis {
        Axis::Rows => idct_along_rows(input, norm),
        Axis::Columns => idct_along_columns(input, norm),
        // FIX #2: The inverse applies axes in the *opposite* order to the forward
        // transform (columns first, then rows).  This is intentional and correct
        // for a separable transform — do NOT "fix" this to match forward order.
        Axis::Both => {
            let temp = idct_along_columns(input, norm);
            idct_along_rows(&temp, norm)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    const EPS: f64 = 1e-9;

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Round-trip: idct(dct(x)) ≈ x  for both norm modes and all axis modes.
    #[test]
    fn roundtrip_ortho_both() {
        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let y = dct_2d_axis(&x, Norm::Ortho, Axis::Both);
        let z = idct_2d_axis(&y, Norm::Ortho, Axis::Both);
        assert!(
            max_abs_diff(&x, &z) < EPS,
            "round-trip error: {}",
            max_abs_diff(&x, &z)
        );
    }

    #[test]
    fn roundtrip_none_both() {
        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let y = dct_2d_axis(&x, Norm::None, Axis::Both);
        let z = idct_2d_axis(&y, Norm::None, Axis::Both);
        assert!(
            max_abs_diff(&x, &z) < EPS,
            "round-trip error: {}",
            max_abs_diff(&x, &z)
        );
    }

    #[test]
    fn roundtrip_ortho_rows() {
        let x = arr2(&[[3.0, 1.0, 4.0, 1.0], [5.0, 9.0, 2.0, 6.0]]);
        let y = dct_2d_axis(&x, Norm::Ortho, Axis::Rows);
        let z = idct_2d_axis(&y, Norm::Ortho, Axis::Rows);
        assert!(max_abs_diff(&x, &z) < EPS);
    }

    #[test]
    fn roundtrip_ortho_columns() {
        let x = arr2(&[[3.0, 1.0], [5.0, 9.0], [2.0, 6.0], [5.0, 3.0]]);
        let y = dct_2d_axis(&x, Norm::Ortho, Axis::Columns);
        let z = idct_2d_axis(&y, Norm::Ortho, Axis::Columns);
        assert!(max_abs_diff(&x, &z) < EPS);
    }

    /// Sanity-check: DC coefficient of Norm::None equals 2·N·mean(row).
    #[test]
    fn dc_coefficient_none() {
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0]]);
        let y = dct_2d_axis(&x, Norm::None, Axis::Rows);
        let expected_dc = 2.0 * x.iter().sum::<f64>(); // 2 · Σ x[n]
        assert!(
            (y[[0, 0]] - expected_dc).abs() < EPS,
            "DC: got {}, expected {}",
            y[[0, 0]],
            expected_dc
        );
    }
}
