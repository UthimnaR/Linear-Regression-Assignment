mod model;
mod train;

use burn::tensor::{Tensor};
use burn_ndarray::NdArray;
use rand::Rng;

fn main() {
    // Generate synthetic data for y = 2x + 1 with noise
    let mut rng = rand::thread_rng();
    let data: Vec<(f32, f32)> = (0..100)
        .map(|x| {
            let x = x as f32;
            let noise = rng.gen_range(-0.1..0.1);
            let y = 2.0 * x + 1.0 + noise;
            (x, y)
        })
        .collect();

    // Create tensors for the input (x) and target (y)
    let inputs: Vec<f32> = data.iter().map(|(x, _)| *x).collect();
    let targets: Vec<f32> = data.iter().map(|(_, y)| *y).collect();

    let inputs_tensor = NdArray::from(inputs);
    let targets_tensor = NdArray::from(targets);

    // Create and train the model
    let mut model = model::LinearRegression::new();
    train::train(&mut model, &inputs_tensor, &targets_tensor, 100, 0.1);

    // After training, test the model on a new input
    let test_input = NdArray::from(vec![5.0]); // Test with x = 5
    let prediction = model.predict(&test_input);
    println!("Prediction for x = 5: {}", prediction);
}
