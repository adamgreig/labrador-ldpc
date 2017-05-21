#![cfg_attr(not(test), no_std)]

pub mod codes;
pub mod encoder;
pub use codes::{LDPCCode};
