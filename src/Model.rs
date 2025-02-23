
use burn::prelude::{Tensor, TensorData, Module};
use burn_ndarray::{NdArray, NdArrayDevice};

#[derive(Module, Debug, Clone)]
pub struct LinearRegression {
    weights: Tensor<NdArray, 1>,
    bias: Tensor<NdArray, 1>,
}

impl LinearRegression {
    pub fn new() -> Self {
        // Create a device for NdArray computation (e.g., CPU).
        let device = NdArrayDevice::default();

        // Initialize weights and bias with the device.
        let weights = Tensor::from_floats([2.2], &device); // Provide the device
        let bias = Tensor::from_floats([0.0], &device);    // Provide the device

        Self { weights, bias }
    }

    pub fn forward(&self, x: Tensor<NdArray, 1>) -> Tensor<NdArray, 1> {
        self.weights.clone() * x + self.bias.clone()
    }
}