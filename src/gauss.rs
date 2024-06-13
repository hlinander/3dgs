// use core::slice::SlicePattern;

use candle_core::{DType, Device, IndexOp};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use duckdb::{params, Connection, Result};
use image::{io::Reader as ImageReader, ImageBuffer, Luma};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use rand::Rng;
use std::{io::Cursor, process::exit};

#[derive(Debug)]
struct Splats {
    splats: Vec<Splat>,
}

impl Splats {
    fn new(vb: &VarBuilder, n_splats: usize) -> Self {
        Self {
            splats: (0..n_splats)
                .map(|idx| Splat::new(&vb.pp(format!("splat{}", idx))))
                .collect(),
        }
    }

    fn render(
        &self,
        device: &Device,
        resolution: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        let mut screen = candle_core::Tensor::zeros((resolution, resolution), DType::F32, device);
        for splat in &self.splats {
            // println!("{}", splat.color);
            screen = screen + splat.render(device, resolution)?;
        }
        screen
    }
    fn render_sparse(
        &self,
        device: &Device,
        resolution: usize,
        fraction: f32,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        let indices = create_sparse_indices(resolution, fraction, device)?;
        let mut screen =
            candle_core::Tensor::zeros((indices.shape().dims()[0],), DType::F32, device);
        for splat in &self.splats {
            // println!("{}", splat.color);
            screen = screen + splat.render_sparse(device, resolution, &indices)?;
        }
        Ok((screen?, indices))
    }
}

#[derive(Debug)]
struct Splat {
    cov: Cov2,
    pos: candle_core::Tensor,
    color: candle_core::Tensor,
}

impl Splat {
    fn new(vb: &VarBuilder) -> Self {
        Self {
            pos: vb
                .get_with_hints(&[2], "pos", candle_nn::init::DEFAULT_KAIMING_NORMAL)
                // .get_with_hints(&[2], "pos", candle_nn::init::ZERO)
                .unwrap(),
            cov: Cov2::new(vb),
            color: vb
                .get_with_hints((1,), "color", candle_nn::init::ZERO)
                .unwrap(),
        }
    }

    // fn sample(&self, device: &Device) -> Vec<f32> {
    // let mat = self.cov.cov(device).unwrap();
    // Vec::new()
    // }

    fn render(
        &self,
        device: &Device,
        resolution: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        let x = candle_core::Tensor::arange(0.0, resolution as f32, device)?;
        let one_over_res = 1.0 / resolution as f64;
        let x = (2.0 * ((one_over_res * x)? - 0.5)?)?;
        let y = candle_core::Tensor::arange(0.0, resolution as f32, device)?;
        let one_over_res = 1.0 / resolution as f64;
        let y = (2.0 * ((one_over_res * y)? - 0.5)?)?;
        let screen = candle_core::Tensor::meshgrid(&[&x, &y], true)?;
        let screen = candle_core::Tensor::stack(screen.as_slice(), 2)?;
        // println!("{}", screen);
        // W, H, 2
        let pos = self.pos.unsqueeze(0)?.unsqueeze(0)?;
        let dist = (screen.expand(&[resolution, resolution, 2])?
            - pos.broadcast_as(&[resolution, resolution, 2]))?
        .contiguous()?;
        // print("{}", )
        let C = self.cov.cov(device)?;
        let det = (C.i((1, 1))? * C.i((0, 0))?)? - (C.i((0, 1))? * C.i((1, 0))?)?;
        let d2 = dist.reshape(((), 2, 1))?;
        let det = det?.unsqueeze(0)?.expand((d2.shape().dims()[0]))?;
        let det = det.unsqueeze(1)?.unsqueeze(1)?;
        let c2 = C.unsqueeze(0)?.broadcast_as((d2.shape().dims()[0], 2, 2))?;
        let c3 = c2.broadcast_matmul(&d2)?;
        let c4 = (-1.0 / 2.0 * d2.t()?.matmul(&c3)?)?;
        let c5 = c4.exp()?;
        // let c5 = (c4.exp()? * det.sqrt()?)? / (2.0 * 3.14159);
        let c6 = c5.reshape((resolution, resolution))?;
        // let c6 = (&self.color.broadcast_as(c6.shape())? * c6)?;
        let c6 = (&candle_nn::ops::sigmoid(&self.color.broadcast_as(c6.shape())?)? * c6)?;
        // println!("{}", c6);
        Ok(c6)
    }
    fn render_sparse(
        &self,
        device: &Device,
        resolution: usize,
        indices: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        let x = candle_core::Tensor::arange(0.0, resolution as f32, device)?;
        let one_over_res = 1.0 / resolution as f64;
        let x = (2.0 * ((one_over_res * x)? - 0.5)?)?;
        let y = candle_core::Tensor::arange(0.0, resolution as f32, device)?;
        let one_over_res = 1.0 / resolution as f64;
        let y = (2.0 * ((one_over_res * y)? - 0.5)?)?;
        let screen = candle_core::Tensor::meshgrid(&[&x, &y], true)?;
        let screen = candle_core::Tensor::stack(screen.as_slice(), 2)?;

        let screen = screen
            .reshape(((), 2))?
            .index_select(&indices, 0)?
            .contiguous()?;

        // println!("{}", screen);
        // W, H, 2
        let pos = self.pos.unsqueeze(0)?;
        let dist = (&screen - pos.broadcast_as(&[screen.shape().dims()[0], 2]))?.contiguous()?;
        // print("{}", )
        let C = self.cov.cov(device)?;
        let det = (C.i((1, 1))? * C.i((0, 0))?)? - (C.i((0, 1))? * C.i((1, 0))?)?;
        let dist = dist.reshape(((), 2, 1))?;
        // let d2 = dist.reshape(((), 2, 1))?;
        let det = det?.unsqueeze(0)?.expand((dist.shape().dims()[0]))?;
        let det = det.unsqueeze(1)?.unsqueeze(1)?;
        let x = C
            .unsqueeze(0)?
            .broadcast_as((dist.shape().dims()[0], 2, 2))?;
        let x = x.broadcast_matmul(&dist)?;
        let x = (-1.0 / 2.0 * dist.t()?.matmul(&x)?)?;
        let x = x.exp()?;
        // let x = (x.exp()? * det.sqrt()?)? / (2.0 * 3.14150);
        let x = x.reshape((indices.shape().dims()[0],))?;
        // let c6 = c5.reshape((resolution, resolution))?;
        // let c6 = (&self.color.broadcast_as(c6.shape())? * c6)?;
        let x = (&candle_nn::ops::sigmoid(&self.color.broadcast_as(x.shape())?)? * x)?;
        // println!("{}", x);
        Ok(x)
    }
}

fn create_sparse_indices(
    resolution: usize,
    fraction: f32,
    device: &Device,
) -> Result<candle_core::Tensor, candle_core::Error> {
    let mut rb = rand::thread_rng();
    let indices: Vec<_> = (0..(resolution * resolution) as i64)
        .filter(|_| rb.gen::<f32>() < fraction)
        .collect();
    let n_pixels = indices.len();
    let indices = candle_core::Tensor::from_vec(indices, (n_pixels,), device)?;
    Ok(indices)
}

#[derive(Debug)]
struct Cov2 {
    rot: QuatRot,
    scale: candle_core::Tensor,
}

impl Cov2 {
    fn new(vb: &VarBuilder) -> Self {
        Self {
            scale: vb
                .get_with_hints(&[2], "scale", candle_nn::init::Init::Const(3.0))
                // .get_with_hints(&[2], "scale", candle_nn::init::DEFAULT_KAIMING_NORMAL)
                .unwrap(),
            rot: QuatRot::new(vb),
        }
    }

    fn cov(&self, device: &Device) -> candle_core::Result<candle_core::Tensor> {
        let R = self.rot.mat3(device)?;
        // println!("R^T*R: {}", R.t()?.matmul(&R)?);
        let m = candle_core::Tensor::eye(2, DType::F32, device)?;
        let scale_shift = (&self.scale);
        let scale = (m * scale_shift.unsqueeze(0)?.expand(&[2, 2]))?;
        // println!("{}", scale);
        R.matmul(&(&scale * &scale)?.matmul(&R.t()?)?)
        // m.i((0, 0)) = 0.0;
    }
}

#[derive(Debug)]
struct QuatRot {
    quat: candle_core::Tensor,
}

impl QuatRot {
    fn new(vb: &VarBuilder) -> Self {
        Self {
            quat: vb
                .get_with_hints(&[4], "quat", candle_nn::init::DEFAULT_KAIMING_NORMAL)
                .unwrap(),
        }
    }
    fn mat3(&self, device: &Device) -> candle_core::Result<candle_core::Tensor> {
        // let r = self.quat.i(0)?;
        // v_i = m_ijk Q^j Q^kj
        let two_s = (2.0 / (&self.quat * &self.quat)?.sum(0)?)?;
        let two_s = two_s.expand(&[9])?;
        // println!("dtype two_s: {:?}", two_s.dtype());
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
        )?;
        // println!("dtype M: {:?}", M.dtype());
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
        // let M2d =
        let quat = M2.matmul(&self.quat.unsqueeze(1)?)?;
        let Mqq = M
            .broadcast_matmul(
                &quat
                    // &self
                    // .quat
                    // .unsqueeze(1)?
                    .unsqueeze(0)?
                    .broadcast_as(&[9, 4, 1])?,
            )?
            .transpose(1, 2)?
            .broadcast_matmul(
                // &self
                // .quat
                &quat
                    // .unsqueeze(1)?
                    .unsqueeze(0)?
                    .broadcast_as(&[9, 4, 1])?,
            )?
            .squeeze(1)?
            .squeeze(1)?;
        // .broadcast_matmul(&self.quat.unsqueeze(1)?)?;
        // println!("Mqq shape: {:?}", Mqq.shape());
        let two_s_M = (two_s * Mqq)?;
        let lhs =
            candle_core::Tensor::new(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], device);
        // let lhs = lhs?.unsqueeze(1)?.unsqueeze(1)?.expand(&[9, 4, 4])?;
        let x = (lhs - ((-1.0) * two_s_M)?)?;
        let x = x.reshape(&[3, 3])?;
        // let ort = x.matmul(&x.t()?)?;
        // println!("ort: {}", x);
        let proj = candle_core::Tensor::new(&[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]], device)?;
        let x = proj.matmul(&x.matmul(&proj.t()?)?)?;
        // let ort = x.matmul(&x.t()?)?;
        // println!("ort2: {}", x);
        // println!("proj: {:?}", x.shape());
        // | 1 0 0 | | a b c |   | a b c |
        // | 0 1 0 | | d e f | = | d e f |
        //           | g h i |

        // | a b c | | 1 0 |   | a b |
        // | d e f | | 0 1 | = | d e |
        //           | 0 0 |
        Ok(x)
    }
}

fn test3_gauss(device: &Device) -> candle_core::Result<()> {
    let conn = Connection::open("duck.db").unwrap();
    conn.execute_batch(
        "
            CREATE SEQUENCE IF NOT EXISTS s;
            CREATE SEQUENCE IF NOT EXISTS t;
            CREATE TABLE IF NOT EXISTS grads
                (
                    id INTEGER DEFAULT nextval('s'),
                    epoch INTEGER,
                    name VARCHAR,
                    value DOUBLE
                );
            CREATE TABLE IF NOT EXISTS tensors
                (
                    id INTEGER DEFAULT nextval('t'),
                    epoch INTEGER,
                    name VARCHAR,
                    value DOUBLE
                );
        ",
    )
    .expect("");
    let varmap: VarMap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let varmap2: VarMap = VarMap::new();
    let vs2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
    // let splat = Splat::new(&vs);
    let splats = Splats::new(&vs.pp("splats"), 10);
    let target_splats = Splats::new(&vs.pp("target"), 3);
    // let splat2 = Splat::new(&vs2);

    let mut sgd = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        },
    )?;

    let res = 32usize;
    let img = ImageReader::open("simple.png")?.decode().unwrap();
    let img = img.resize(res as u32, res as u32, image::imageops::FilterType::Nearest);
    let imgf32 = img.grayscale().into_rgb32f();
    let pixels: Vec<f32> = imgf32.pixels().map(|pixel| pixel[0]).collect();
    let target = candle_core::Tensor::from_vec(pixels, (res, res), device)?.detach();
    // println!("{}", target);
    // let pixels_gray = pixels.into_iter().map(|pixel| pixel[0])
    // println!("{}", pixels.len());
    // exit(0);
    // println!("{:?}", target_splats);
    // let target = target_splats.render(device, res)?.detach();
    let target_flat = target.flatten_all()?;
    // println!("{:?}", splat2);
    save_image(&target, "target.png", res);
    let mut idx = 0usize;
    loop {
        // let screen = splats.render(device, res)?;
        let (screen, indices) = splats.render_sparse(device, res, 0.5)?;
        let target_pixels = target_flat.index_select(&indices, 0)?;

        // let loss = candle_nn::loss::mse(&screen, &target)?;
        let loss = candle_nn::loss::mse(&screen, &target_pixels)?;
        let grads = loss.backward()?;
        // println!("pos: {:?}", gradps);
        // println!("scales: {:?}", gradscales);
        // let p_grad = grads.get(&splats.splats[0].pos).unwrap();
        // println!("{}", p_grad);
        sgd.step(&grads);
        // sgd.backward_step(&loss)?;

        if idx % 50 == 0 {
            let gradps: Vec<_> = splats
                .splats
                .iter()
                .map(|splat| {
                    let gp = grads.get(&splat.pos).unwrap();
                    let gp = (gp * gp).unwrap().sum_all().unwrap();
                    gp.to_scalar::<f32>().unwrap()
                })
                .sorted_by(|a, b| a.partial_cmp(b).expect(format!("{} {}", a, b).as_str()))
                .rev()
                .collect();

            log_grads(&splats, &grads, &conn, idx);
            println!("{}", loss.to_scalar::<f32>()?);
            let screen = splats.render(device, res)?;
            let error = (&screen - &target).unwrap().abs()?;
            let min = error
                .flatten_all()?
                .min(0)?
                .unsqueeze(0)?
                .unsqueeze(0)?
                .expand((res, res))?;
            let max = error
                .flatten_all()?
                .max(0)?
                .unsqueeze(0)?
                .unsqueeze(0)?
                .expand((res, res))?;
            let error = ((error - &min) / (&max - &min)?)?;
            save_image(&error, "error.png", res);
            save_image(&screen, "output.png", res);
        }
        idx += 1;
        // println!("pos: {}", splat.pos);
        // println!("target: {}", splat2.pos);
    }

    // let loss =
    // println!("{}", screen);
    Ok(())
}

fn log_grads(
    splats: &Splats,
    grads: &candle_core::backprop::GradStore,
    conn: &Connection,
    idx: usize,
) {
    let gradps: Vec<_> = splats
        .splats
        .iter()
        .map(|splat| {
            let gp = grads.get(&splat.pos).unwrap();
            let gp = (gp * gp).unwrap().sum_all().unwrap();
            gp.to_scalar::<f32>().unwrap()
        })
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .rev()
        .collect();
    let gradscales: Vec<_> = splats
        .splats
        .iter()
        .map(|splat| {
            let gp = grads.get(&splat.cov.scale).unwrap();
            let gp = (gp * gp).unwrap().sum_all().unwrap();
            gp.to_scalar::<f32>().unwrap()
        })
        .sorted_by(|a, b| a.partial_cmp(b).expect(format!("{} {}", a, b).as_str()))
        .rev()
        .collect();
    for v in gradscales {
        conn.execute(
            "INSERT INTO grads (epoch, name, value) VALUES (?, ?, ?)",
            params![idx, "scale", v],
        )
        .expect("");
    }
    for v in gradps {
        conn.execute(
            "INSERT INTO grads (epoch, name, value) VALUES (?, ?, ?)",
            params![idx, "pos", v],
        )
        .expect("");
    }
    for splat in &splats.splats {
        conn.execute(
            "INSERT INTO tensors (epoch, name, value) VALUES (?, ?, ?)",
            params![idx, "color", splat.color.to_vec1::<f32>().unwrap()[0]],
        )
        .expect("");
    }
}

fn save_image(screen: &candle_core::Tensor, name: &str, res: usize) -> candle_core::Result<()> {
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
fn main() -> candle_core::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    test3_gauss(&dev)?;
    Ok(())
}
