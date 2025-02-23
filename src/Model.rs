
use burn::tensor::{Tensor};
use burn_ndarray::NdArray;

pub struct LinearRegression {
    weights: Tensor<NdArray, f32>,
    bias: Tensor<NdArray, f32>,
}

impl LinearRegression {
    pub fn new() -> Self {
        let weights = Tensor::from(NdArray::from(vec![0.0]));  // Initialize weights to 0
        let bias = Tensor::from(NdArray::from(vec![0.0]));     // Initialize bias to 0
        LinearRegression { weights, bias }
    }

    pub fn forward(&self, inputs: &NdArray<f32>) -> NdArray<f32> {
        inputs * &self.weights + &self.bias
    }

    pub fn predict(&self, inputs: &NdArray<f32>) -> f32 {
        self.forward(inputs).into_inner()[0]
    }
}
