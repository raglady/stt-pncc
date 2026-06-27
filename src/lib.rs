cfg_select! {
    feature = "f32" => {
        mod F32;
        pub use F32::*;
    }
    feature = "f64" => {
        mod F64;
        pub use F64::*;
    }
    _ => {}
}
