use burn_ndarray::NdArray;
use burn::{
    tensor::{Tensor},
    optim::{Adam as OtherAdam, Optimizer},
};

use crate::model::LinearRegression;

struct Adam<'a>(&'a LinearRegression, f32);

impl<'a> Adam<'a> {
    fn new(p0: &LinearRegression, p1: f32) -> _ {
        todo!()
    }
}

struct Tensor(Vec<f32>);

pub fn train_model(x_train: Vec<f32>, y_train: Vec<f32>, epochs: usize, lr: f32) -> LinearRegression {
    let mut model = LinearRegression::new();
    let mut optimizer = Adam::new(&model, lr);

    let x_train_tensor = Tensor::<NdArray, 1>::from_floats(x_train);
    let y_train_tensor = Tensor::<NdArray, 1>::from_floats(y_train);

    for epoch in 0..epochs {
        let y_pred = model.forward(x_train_tensor.clone());
        let loss = (y_pred.clone() - y_train_tensor.clone()).powf(2.0).mean(); // Mean Squared Error

        optimizer.update(&mut model, loss.clone());

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.into_scalar());
        }
    }

    model
}

fn NdArray(p0: Vec<f32>) -> _ {
    todo!()
}
