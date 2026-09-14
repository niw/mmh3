//! Support shared by the generation and diagnostic command-line programs.

pub mod cli;
pub mod output;

#[cfg(feature = "cuda")]
pub mod models;
