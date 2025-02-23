use burn::tensor::{Tensor};
use burn_ndarray::NdArray;
use burn::optimizer::{SGD};
use burn::ops::{Add, Mul, Mean};

// Mean Squared Error Loss function
fn mean_squared_error(pred: &NdArray<f32>, target: &NdArray<f32>) -> NdArray<f32> {
    (pred - target).powi(2).mean().reshape([1])  // MSE = (y_pred - y_true)^2
}

pub fn train(model: &mut super::model::LinearRegression, inputs: &NdArray<f32>, targets: &NdArray<f32>, epochs: usize, learning_rate: f32) {
    let mut optimizer = SGD::default().learning_rate(learning_rate);

    for epoch in 0..epochs {
        // Forward pass
        let predictions = model.forward(inputs);

        // Compute loss
        let loss = mean_squared_error(&predictions, targets);

        // Backward pass
        loss.backward();

        // Update weights and bias
        optimizer.step(&mut model.weights);
        optimizer.step(&mut model.bias);

        // Print loss every 10 epochs
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss);
        }
    }
}
