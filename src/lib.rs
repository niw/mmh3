//! Support shared by the generation and diagnostic command-line programs.

pub mod audio;
pub mod cli;
pub mod output;
pub mod pictures;

#[cfg(feature = "cuda")]
pub mod generation;
#[cfg(feature = "cuda")]
pub mod models;
#[cfg(feature = "cuda")]
pub mod video;
