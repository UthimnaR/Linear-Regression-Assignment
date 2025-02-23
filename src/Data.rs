use rand::Rng;

pub fn generate_data(samples: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..samples {
        let x: f32 = rng.gen_range(-10.0..10.0);
        let noise: f32 = rng.gen_range(-1.0..1.0); // Add noise
        let y: f32 = 2.0 * x + 1.0 + noise;

        x_data.push(x);
        y_data.push(y);
    }

    (x_data, y_data)
}
