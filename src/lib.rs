//! Support shared by the generation and diagnostic command-line programs.

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("choose one GPU backend: cuda or metal");
#[cfg(all(feature = "metal", not(target_os = "macos")))]
compile_error!("the metal feature requires macOS");

pub mod audio;
pub mod cli;
pub mod output;
pub mod pictures;

pub mod generate;
pub mod generation;
#[cfg(feature = "metal")]
pub mod metal;
pub mod models;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod resident;
#[cfg(feature = "server")]
pub mod server;
pub mod video;
pub mod worker;
