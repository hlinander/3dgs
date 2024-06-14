#[cfg(feature = "metal")]
extern crate accelerate_src;

use candle_core::{DType, Device};
use candle_nn::{Optimizer, VarBuilder, VarMap};
#[cfg(feature = "duckdb")]
use duckdb::{Connection, Result};
use image::{io::Reader as ImageReader, ImageBuffer, Luma};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use rand::Rng;
use std::time::{Duration, Instant};
use std::{io::Cursor, process::exit};

#[cfg(feature = "duckdb")]
mod db;

struct FastSplats {
    pos: candle_core::Tensor,
    color: candle_core::Tensor,
    scale: candle_core::Tensor,
    quat: candle_core::Tensor,
    screen: candle_core::Tensor,
    quat_constructor: candle_core::Tensor,
    n_splats: usize,
    resolution: usize,
}

impl FastSplats {
    fn new(
        vb: &VarBuilder,
        n_splats: usize,
        resolution: usize,
        scale: f64,
        device: &Device,
    ) -> Self {
        let x = candle_core::Tensor::arange(0.0, resolution as f32, device).unwrap();
        let one_over_res = 1.0 / resolution as f64;
        let x = (2.0 * ((one_over_res * x).unwrap() - 0.5).unwrap()).unwrap();
        let y = candle_core::Tensor::arange(0.0, resolution as f32, device).unwrap();
        let one_over_res = 1.0 / resolution as f64;
        let y = (2.0 * ((one_over_res * y).unwrap() - 0.5).unwrap()).unwrap();
        let screen = candle_core::Tensor::meshgrid(&[&x, &y], true).unwrap();
        let screen = candle_core::Tensor::stack(screen.as_slice(), 2).unwrap();
        let screen = screen
            .unsqueeze(0)
            .unwrap()
            .expand(&[n_splats, resolution, resolution, 2])
            .unwrap();
        let M = candle_core::Tensor::new(
            &[
                [
                    // r,   i,   j,   k
                    [0.0f32, 0.0, 0.0, 0.0], // r
                    [0.0, 0.0, 0.0, 0.0],    // i
                    [0.0, 0.0, -1.0, 0.0],   // j
                    [0.0, 0.0, 0.0, -1.0],   // k
                ],
                [
                    // r,   i,   j,   k
                    [0.0, 0.0, 0.0, -1.0], // r
                    [0.0, 0.0, 0.0, 0.0],  // i
                    [0.0, 1.0, 0.0, 0.0],  // j
                    [0.0, 0.0, 0.0, 0.0],  // k
                ],
                [
                    // r,   i,   j,   k
                    [0.0, 0.0, 1.0, 0.0], // r
                    [0.0, 0.0, 0.0, 0.0], // i
                    [0.0, 0.0, 0.0, 0.0], // j
                    [0.0, 1.0, 0.0, 0.0], // k
                ],
                [
                    // r,   i,   j,   k
                    [0.0, 0.0, 0.0, 1.0], // r
                    [0.0, 0.0, 0.0, 0.0], // i
                    [0.0, 1.0, 0.0, 0.0], // j
                    [0.0, 0.0, 0.0, 0.0], // k
                ],
                [
                    // r,   i,   j,   k
                    [0.0, 0.0, 0.0, 0.0],  // r
                    [0.0, -1.0, 0.0, 0.0], // i
                    [0.0, 0.0, 0.0, 0.0],  // j
                    [0.0, 0.0, 0.0, -1.0], // k
                ],
                [
                    // r,   i,   j,   k
                    [0.0, -1.0, 0.0, 0.0], // r
                    [0.0, 0.0, 0.0, 0.0],  // i
                    [0.0, 0.0, 0.0, 0.0],  // j
                    [0.0, 0.0, 1.0, 0.0],  // k
                ],
                [
                    // r,   i,   j,   k
                    [0.0, 0.0, -1.0, 0.0], // r
                    [0.0, 0.0, 0.0, 0.0],  // i
                    [0.0, 0.0, 0.0, 0.0],  // j
                    [0.0, 1.0, 0.0, 0.0],  // k
                ],
                [
                    // r,   i,   j,   k
                    [0.0, 1.0, 0.0, 0.0], // r
                    [0.0, 0.0, 0.0, 0.0], // i
                    [0.0, 0.0, 0.0, 0.0], // j
                    [0.0, 0.0, 1.0, 0.0], // k
                ],
                [
                    // r,   i,   j,   k
                    [0.0, 0.0, 0.0, 0.0],  // r
                    [0.0, -1.0, 0.0, 0.0], // i
                    [0.0, 0.0, -1.0, 0.0], // j
                    [0.0, 0.0, 0.0, 0.0],  // k
                ],
            ],
            device,
        )
        .unwrap();

        Self {
            resolution,
            quat_constructor: M,
            screen,
            n_splats,
            pos: vb
                .get_with_hints(
                    &[n_splats, 2],
                    "pos",
                    candle_nn::init::DEFAULT_KAIMING_NORMAL,
                )
                .unwrap(),
            color: vb
                .get_with_hints(
                    &[n_splats, 1],
                    "color",
                    // candle_nn::init::DEFAULT_KAIMING_NORMAL,
                    candle_nn::init::Init::Const(0.1),
                )
                .unwrap(),
            scale: vb
                .get_with_hints(&[n_splats, 2], "scale", candle_nn::init::Init::Const(scale))
                .unwrap(),

            quat: vb
                .get_with_hints(
                    &[n_splats, 4],
                    "quat",
                    candle_nn::init::DEFAULT_KAIMING_NORMAL,
                )
                .unwrap(),
        }
    }

    fn rot(&self, device: &Device) -> candle_core::Result<candle_core::Tensor> {
        let M2 = candle_core::Tensor::new(
            &[
                // r,   i,   j,   k
                [1.0f32, 0.0, 0.0, 0.0], // r
                [0.0, 0.0, 0.0, 0.0],    // i
                [0.0, 0.0, 0.0, 0.0],    // j
                [0.0, 0.0, 0.0, 1.0],    // k
            ],
            device,
        )?;
        let quat = M2.broadcast_matmul(&self.quat.unsqueeze(2)?)?;
        let two_s = (2.0 / (&quat * &quat)?.sum(1)?)?;
        let two_s = two_s.expand(&[self.n_splats, 9])?;
        let bquat = quat
            .unsqueeze(1)?
            .broadcast_as(&[self.n_splats, 9, 4, 1])?
            .contiguous()?;
        let Mqq = self
            .quat_constructor
            .unsqueeze(0)?
            .broadcast_as(&[self.n_splats, 9, 4, 4])?
            .contiguous()?;
        let Mqqmat = Mqq.matmul(&bquat)?;
        let Mqqmat2 = Mqqmat
            .transpose(2, 3)?
            .matmul(
                &quat
                    .unsqueeze(1)?
                    .broadcast_as(&[self.n_splats, 9, 4, 1])?
                    .contiguous()?,
            )?
            .squeeze(2)?
            .squeeze(2)?;

        let two_s_M = (two_s * Mqqmat2)?;
        let lhs =
            candle_core::Tensor::new(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], device);
        let x = (lhs?.unsqueeze(0)?.expand(&[self.n_splats, 9]) - ((-1.0) * two_s_M)?)?;
        let x = x.reshape(&[self.n_splats, 3, 3])?;
        let proj = candle_core::Tensor::new(&[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]], device)?;
        let proj = proj
            .unsqueeze(0)?
            .expand(&[self.n_splats, 2, 3])?
            .contiguous()?;
        let x = proj.matmul(&x.matmul(&proj.t()?)?)?;
        Ok(x)
    }

    fn cov(&self, device: &Device) -> candle_core::Result<candle_core::Tensor> {
        let R = self.rot(device)?;
        let m = candle_core::Tensor::eye(2, DType::F32, device)?;
        let scale_shift = candle_core::Tensor::exp(&self.scale)?;
        let scale = (m.unsqueeze(0)?.expand(&[self.n_splats, 2, 2])?
            * scale_shift.unsqueeze(1)?.expand(&[self.n_splats, 2, 2]))?;
        let scale = scale.contiguous()?;
        R.matmul(&(&scale * &scale)?.matmul(&R.t()?)?)
    }

    fn render(
        &self,
        // resolution: usize,
        device: &Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        let resolution = self.resolution;
        let pos = self.pos.unsqueeze(1)?.unsqueeze(1)?;
        let dist = (&self.screen - pos.broadcast_as(&[self.n_splats, resolution, resolution, 2]))?
            .contiguous()?;
        let C = self.cov(device)?;
        let d2 = dist.reshape((self.n_splats, (), 2, 1))?;
        let c2 = C
            .unsqueeze(1)?
            .broadcast_as((self.n_splats, d2.shape().dims()[1], 2, 2))?
            .contiguous()?;
        let c3 = c2.matmul(&d2)?;
        let c4 = (-1.0 / 2.0 * d2.t()?.matmul(&c3)?)?;
        let c5 = c4.exp()?;
        let c6 = c5.reshape((self.n_splats, resolution, resolution))?;
        let color = self
            .color
            .unsqueeze(1)?
            .expand(&[self.n_splats, resolution, resolution])?;
        let c6 = (candle_core::Tensor::exp(&color) * c6)?;
        // let c6 = (candle_nn::ops::sigmoid(&color)? * c6)?;
        let c6 = c6.sum(0)?;
        Ok(c6)
    }
}

fn train_splats(
    target: candle_core::Tensor,
    n_splats_per_batch: usize,
    n_batches: usize,
    initial_scale: f64,
    epochs: usize,
    device: &Device,
) -> candle_core::Result<Vec<FastSplats>> {
    let res = target.shape().dims()[0];
    let varmap: VarMap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let splats = (0..n_batches)
        .map(|idx| {
            FastSplats::new(
                &vs.pp(format!("batch_{}", idx)),
                n_splats_per_batch,
                res,
                initial_scale,
                device,
            )
        })
        .collect_vec();
    // let splats = FastSplats::new(&vs, n_splats, res, initial_scale, device);

    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        },
    )?;
    #[cfg(feature = "duckdb")]
    let db_connection = db::init_db();
    let mut next_log = Instant::now();
    for epoch in 0..epochs {
        for active_idx in 0..n_batches {
            let mut screen = candle_core::Tensor::zeros((res, res), DType::F32, device)?;
            for (batch_idx, batch) in splats.iter().enumerate() {
                let batch_screen = batch.render(device)?;
                if batch_idx == active_idx {
                    screen = (screen + batch_screen)?;
                } else {
                    screen = (screen + batch_screen.detach())?;
                }
            }
            let screen = candle_nn::ops::sigmoid(&screen)?;
            let loss = candle_nn::loss::mse(&screen, &target)?;
            let grads = loss.backward()?;
            optimizer.step(&grads)?;
            if Instant::now().duration_since(next_log).as_millis() > 0 {
                next_log = Instant::now() + Duration::from_secs(2);
                println!("loss: {}", loss.detach().to_scalar::<f32>().unwrap());
                #[cfg(feature = "duckdb")]
                log_gradients(grads, &splats, &db_connection, epoch)?;
                save_image(&screen, "output.png")?;
            }
        }
    }

    Ok(splats)
}

#[cfg(feature = "duckdb")]
fn log_gradients(
    grads: candle_core::backprop::GradStore,
    splats: &Vec<FastSplats>,
    db_connection: &Connection,
    epoch: usize,
) -> Result<(), candle_core::Error> {
    for splat in splats {
        if let Some(gp) = grads.get(&splat.pos) {
            let gp = (gp * gp)?.sum(1)?.to_vec1::<f32>()?;
            db::log_grad(db_connection, gp, "pos".to_string(), epoch);
        }
    }
    Ok(())
}

fn save_image(screen: &candle_core::Tensor, name: &str) -> candle_core::Result<()> {
    let res = screen.shape().dims()[0];
    let min = screen
        .flatten_all()?
        .min(0)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .expand((res, res))?;
    let max = screen
        .flatten_all()?
        .max(0)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .expand((res, res))?;
    let screen = ((screen - &min) / (&max - &min)?)?;
    let screen_pixels = screen.to_vec2::<f32>().unwrap();
    let screen_pixels: Vec<f32> = screen_pixels
        .into_iter()
        .flat_map(|x| x.into_iter())
        .collect();

    let mut imgbuf: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(res as u32, res as u32);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let value = (screen_pixels[(y * res as u32 + x) as usize] * 255.0) as u8;
        *pixel = Luma([value]);
    }

    imgbuf.save(name).unwrap();
    Ok(())
}

fn load_image(path: &str, res: usize, device: &Device) -> candle_core::Result<candle_core::Tensor> {
    let img = ImageReader::open(path)?.decode().unwrap();
    let img = img.resize_exact(res as u32, res as u32, image::imageops::FilterType::Nearest);
    let imgf32 = img.grayscale().into_rgb32f();
    let pixels: Vec<f32> = imgf32.pixels().map(|pixel| pixel[0]).collect();
    // println!("{}", pixels.len());
    let target = candle_core::Tensor::from_vec(pixels, (res, res), device)?.detach();
    Ok(target)
}

fn generate_target_from_splats(device: &Device) -> Result<candle_core::Tensor, candle_core::Error> {
    let varmap: VarMap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let target_splats = FastSplats::new(&vs, 50, 128, 6.0, device);
    let target = target_splats.render(device)?.detach();
    Ok(target)
}

fn main() -> candle_core::Result<()> {
    // let dev = candle_core::Device::cuda_if_available(0)?;
    // let dev = candle_core::Device::new_metal(0)?;
    let device = match candle_core::Device::new_cuda(0) {
        Ok(device) => device,
        Err(_) => match candle_core::Device::new_metal(0) {
            Ok(device) => device,
            Err(_) => candle_core::Device::Cpu,
        },
    };
    println!("{:?}", device);

    // let target = generate_target_from_splats(&device)?;
    let target = load_image("deepfield.png", 64, &device)?;
    save_image(&target, "target.png")?;
    let trained_splats = train_splats(target, 100, 1, 3.0, 100000, &device)?;
    Ok(())
}
