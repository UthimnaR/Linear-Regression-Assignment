mod model;
mod train;
mod data;

use burn_ndarray::NdArray;
use textplots::{Chart, Plot, Shape};

fn main() {
    let (x_train, y_train) = data::generate_data(100);

    let model = train::train_model(x_train.clone(), y_train.clone(), 100, 0.01);

    let x_test: Vec<f32> = (-10..10).map(|x| x as f32).collect();
    let x_test_tensor = burn::tensor::Tensor::<NdArray, 1>::from_floats(x_test.clone());
    let y_pred = model.forward(x_test_tensor).into_data().value;

    // Plot results
    println!("Model Predictions:");
    Chart::new(100, 30, -10.0, 10.0)
        .lineplot(&Shape::Lines(&x_test, &y_pred))
        .display();
}
