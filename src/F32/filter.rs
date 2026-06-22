/// Applies a rational IIR/FIR filter to `signal`.
///
/// Implements the difference equation:
///   a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... - a[1]*y[n-1] - a[2]*y[n-2] - ...
///
/// # Panics
/// - If `denominator` is empty or `a[0]` is zero
/// - If `numerator` is empty
/// - If any input coefficient or signal value is NaN or infinite
pub fn filter(numerator: &[f32], denominator: &[f32], signal: &[f32]) -> Vec<f32> {
    // --- Validation ---
    assert!(
        !denominator.is_empty() && denominator[0] != 0.0,
        "denominator[0] must be non-zero"
    );
    assert!(!numerator.is_empty(), "numerator must not be empty");

    debug_assert!(
        numerator.iter().all(|x| x.is_finite()),
        "numerator contains NaN or Inf"
    );
    debug_assert!(
        denominator.iter().all(|x| x.is_finite()),
        "denominator contains NaN or Inf"
    );

    if signal.is_empty() {
        return Vec::new();
    }

    let n = signal.len();
    let nb = numerator.len();
    let _na = denominator.len();

    // Pre-normalise coefficients by a[0] — avoids repeated division in hot loop
    let a0 = denominator[0];
    let b: Vec<f32> = numerator.iter().map(|&v| v / a0).collect();
    let a: Vec<f32> = denominator.iter().skip(1).map(|&v| v / a0).collect();
    let na_norm = a.len(); // == na - 1

    let mut y = vec![0.0_f32; n];

    for i in 0..n {
        // Numerator (FIR) part
        let b_len = nb.min(i + 1);
        let sum_b: f32 = b[..b_len]
            .iter()
            .zip(signal[..=i].iter().rev())
            .map(|(&bj, &xj)| bj * xj)
            .sum();

        // Denominator (IIR feedback) part
        let a_len = na_norm.min(i);
        let sum_a: f32 = a[..a_len]
            .iter()
            .zip(y[..i].iter().rev())
            .map(|(&aj, &yj)| aj * yj)
            .sum();

        y[i] = sum_b - sum_a;
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // --- FIR: moving average (b=[0.5, 0.5], a=[1.0]) ---
    #[test]
    fn test_fir_moving_average() {
        let b = [0.5, 0.5];
        let a = [1.0];
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = filter(&b, &a, &x);
        // y[0]=0.5, y[1]=1.5, y[2]=2.5, y[3]=3.5
        assert!(approx_eq(&y, &[0.5, 1.5, 2.5, 3.5], 1e-10));
    }

    // --- IIR: simple leaky integrator (b=[1], a=[1, -0.9]) ---
    #[test]
    fn test_iir_leaky_integrator() {
        let b = [1.0];
        let a = [1.0, -0.9];
        let x = [1.0, 0.0, 0.0, 0.0, 0.0];
        let y = filter(&b, &a, &x);
        // y[n] = 0.9^n
        let expected: Vec<f32> = (0..5).map(|n| 0.9_f32.powi(n)).collect();
        assert!(approx_eq(&y, &expected, 1e-10));
    }

    // --- Identity filter (b=[1], a=[1]) ---
    #[test]
    fn test_identity_filter() {
        let x = vec![1.0, -2.0, 3.5, 0.0, 7.1];
        let y = filter(&[1.0], &[1.0], &x);
        assert!(approx_eq(&y, &x, 1e-10));
    }

    // --- Scaling: b=[2], a=[1] doubles signal ---
    #[test]
    fn test_gain_filter() {
        let x = [1.0, 2.0, 3.0];
        let y = filter(&[2.0], &[1.0], &x);
        assert!(approx_eq(&y, &[2.0, 4.0, 6.0], 1e-10));
    }

    // --- a[0] normalisation: a=[2, ...] same as a=[1, ...] ---
    #[test]
    fn test_a0_normalisation() {
        let x = [1.0, 2.0, 3.0];
        let y1 = filter(&[1.0], &[1.0, -0.5], &x);
        let y2 = filter(&[2.0], &[2.0, -1.0], &x);
        assert!(approx_eq(&y1, &y2, 1e-10));
    }

    // --- Empty signal returns empty ---
    #[test]
    fn test_empty_signal() {
        let y = filter(&[1.0], &[1.0], &[]);
        assert!(y.is_empty());
    }

    // --- Panics ---
    #[test]
    #[should_panic(expected = "denominator[0] must be non-zero")]
    fn test_panic_zero_a0() {
        filter(&[1.0], &[0.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "denominator[0] must be non-zero")]
    fn test_panic_empty_denominator() {
        filter(&[1.0], &[], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "numerator must not be empty")]
    fn test_panic_empty_numerator() {
        filter(&[], &[1.0], &[1.0]);
    }
}
