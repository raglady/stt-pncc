# stt-pncc

> Power-Normalized Cepstral Coefficients (PNCC) implementation in Rust — designed for robust speech feature extraction in noisy environments.

[![Crates.io](https://img.shields.io/crates/v/stt-pncc.svg)](https://crates.io/crates/stt-pncc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org)

---

## What is PNCC?

PNCC (Power-Normalized Cepstral Coefficients) is a speech feature extraction method that outperforms classical MFCC in noisy conditions. It applies:

- **Gammatone-like filtering** for auditory-inspired frequency decomposition
- **Power normalization** using a medium-time power estimate, making features robust to additive and convolutive noise
- **Asymmetric noise suppression** to reduce background interference
- **DCT** to produce compact cepstral coefficients ready for ASR models

This crate is part of the broader [`stt`](https://github.com/raglady/stt-decoder) project — a Speech-to-Text pipeline built entirely in Rust.

---

## Features

- Pure Rust implementation — no C/C++ bindings required
- Parallelized via [Rayon](https://github.com/rayon-rs/rayon) for multi-threaded processing
- FFT powered by [RustFFT](https://github.com/ejmahler/RustFFT)
- N-dimensional array support via [ndarray](https://github.com/rust-ndarray/ndarray)
- Choose numeric precision at compile time: `f32` or `f64`

---

## Installation

Add the crate to your `Cargo.toml`. **You must select a precision feature** (`f32` or `f64`) — there is no default:

```toml
[dependencies]
stt-pncc = { version = "1.0.0", features = ["f32"] }
```

Or for double precision:

```toml
[dependencies]
stt-pncc = { version = "1.0.0", features = ["f64"] }
```

> ⚠️ Enabling both features simultaneously is not supported.

---

## Quick Start

```rust
use stt_pncc::Pncc;

fn main() {
    // Raw audio samples at 16kHz
    let audio: Vec<f32> = vec![/* your PCM samples */];

    let pncc = Pncc::new(
        16000,  // sample rate
        40,     // number of filters
        13,     // number of cepstral coefficients
    );

    let features = pncc.compute(&audio);
    println!("PNCC features shape: {:?}", features.shape());
}
```

---

## Architecture

This crate is one component of a modular STT pipeline:

```
Raw Audio
    │
    ▼
┌─────────────┐
│  stt-pncc   │  ← Feature extraction (this crate)
│  PNCC feats │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ stt-decoder │  ← Acoustic model decoding (CUDA / WGSL)
└──────┬──────┘
       │
       ▼
  Transcribed text
```

---

## Dependencies

| Crate | Purpose |
|---|---|
| `ndarray` | N-dimensional array operations with parallel support |
| `rustfft` | Fast Fourier Transform |
| `num-complex` | Complex number arithmetic |
| `rayon` | Data-parallelism for multi-threaded processing |

---

## Feature Flags

| Flag | Description |
|---|---|
| `f32` | Use 32-bit floating point precision (recommended for inference) |
| `f64` | Use 64-bit floating point precision (higher accuracy, more memory) |

---

## Why Rust?

Most PNCC implementations exist in Python (e.g. `spafe`). This implementation targets production STT systems where Python overhead is unacceptable — low-latency transcription, embedded devices, and GPU-adjacent pipelines where data movement must be minimized.

---

## Related Crates

- [`stt-decoder`](https://github.com/raglady/stt-decoder) — Decoder library using CUDA and WGSL for GPU-accelerated inference
- [`stt-train`](https://github.com/raglady/stt-train) — Training pipeline for the STT acoustic model

---

## License

MIT — see [LICENSE](LICENSE) for details.
