#![recursion_limit = "256"]
mod data;
mod inference;
mod model;
mod training;

use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Cuda},
    data::dataloader::Dataset,
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::cuda::CudaDevice::default();
    let artifact_dir = "/tmp/guide";
    // crate::training::train::<MyAutodiffBackend>(
    //     artifact_dir,
    //     TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
    //     device.clone(),
    // );
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
