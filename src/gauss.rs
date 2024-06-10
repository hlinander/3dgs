// use core::slice::SlicePattern;

use candle_core::{DType, Device, IndexOp};
use candle_nn::{Optimizer, VarBuilder, VarMap};

// def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
//     """
//     Convert rotations given as quaternions to rotation matrices.

//     Args:
//         quaternions: quaternions with real part first,
//             as tensor of shape (..., 4).

//     Returns:
//         Rotation matrices as tensor of shape (..., 3, 3).
//     """
//     r, i, j, k = torch.unbind(quaternions, -1)
//     # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
//     two_s = 2.0 / (quaternions * quaternions).sum(-1)

//     o = torch.stack(
//         (
//             1 - two_s * (j * j + k * k),
//             two_s * (i * j - k * r),
//             two_s * (i * k + j * r),
//             two_s * (i * j + k * r),
//             1 - two_s * (i * i + k * k),
//             two_s * (j * k - i * r),
//             two_s * (i * k - j * r),
//             two_s * (j * k + i * r),
//             1 - two_s * (i * i + j * j),
//         ),
//         -1,
//     )
//     return o.reshape(quaternions.shape[:-1] + (3, 3))

struct Splat {
    cov: Cov2,
    pos: candle_core::Tensor,
}

impl Splat {
    fn new(vb: &VarBuilder) -> Self {
        Self {
            pos: vb
                .get_with_hints(&[2], "pos", candle_nn::init::DEFAULT_KAIMING_NORMAL)
                .unwrap(),
            cov: Cov2::new(vb),
        }
    }

    fn render(
        &self,
        device: &Device,
        resolution: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        let x = candle_core::Tensor::arange(0.0, resolution as f32, device)?;
        let y = candle_core::Tensor::arange(0.0, resolution as f32, device)?;
        let screen = candle_core::Tensor::meshgrid(&[&x, &y], true)?;
        let screen = candle_core::Tensor::stack(screen.as_slice(), 2)?;
        // let mut pos = candle_core::Tensor::new(&[[0.0f32, 0.0], [12.0f32, 12.0]], device)?;
        // let mut cov = candle_core::Tensor::new(&[[[1.0, 0.0], [0.0, 1.0]], [[]]], device)?;
        let pos = self.pos.unsqueeze(0)?.unsqueeze(0)?;
        // println!("pos: {}", pos);
        // let B = pos.shape().dims()[0];
        // println!("dist: {:?}", pos.shape());
        // pos = pos.expand(&[0, 12, 12, 2])?;
        let dist = (screen.expand(&[resolution, resolution, 2])?
            - pos.broadcast_as(&[resolution, resolution, 2]))?
        .contiguous()?;
        // let dist = (&dist * &dist)?.sum(3)?;
        // println!("dist: {}", dist);

        // let varmap: VarMap = VarMap::new();
        // let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // let q = QuatRot::new(&vs);
        // let cov = Cov2::new(&vs);
        let C = self.cov.cov(device)?;
        // println!("cov: {}", C.t()?.matmul(&C)?);
        // println!("cont: {}", C.is_contiguous());
        let d2 = dist.reshape(((), 2, 1))?;
        let c2 = C.unsqueeze(0)?.broadcast_as((d2.shape().dims()[0], 2, 2))?;
        let c3 = c2.broadcast_matmul(&d2)?;
        let c4 = (-1.0 * d2.t()?.matmul(&c3)?)?;
        let c5 = c4.exp()?;
        let c6 = c5.reshape((resolution, resolution))?;
        Ok(c6)
    }
}

struct Cov2 {
    rot: QuatRot,
    scale: candle_core::Tensor,
}

impl Cov2 {
    fn new(vb: &VarBuilder) -> Self {
        Self {
            scale: vb
                .get_with_hints(&[2], "scale", candle_nn::init::DEFAULT_KAIMING_NORMAL)
                .unwrap(),
            rot: QuatRot::new(vb),
        }
    }

    fn cov(&self, device: &Device) -> candle_core::Result<candle_core::Tensor> {
        let R = self.rot.mat3(device)?;
        // println!("R^T*R: {}", R.t()?.matmul(&R)?);
        let m = candle_core::Tensor::eye(2, DType::F32, device)?;
        let scale = (m * self.scale.unsqueeze(0)?.expand(&[2, 2]))?;
        R.matmul(&(&scale * &scale)?.matmul(&R.t()?)?)
        // m.i((0, 0)) = 0.0;
    }
}

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
        let ort = x.matmul(&x.t()?)?;
        // println!("ort: {}", x);
        let proj = candle_core::Tensor::new(&[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]], device)?;
        let x = proj.matmul(&x.matmul(&proj.t()?)?)?;
        let ort = x.matmul(&x.t()?)?;
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
struct NQuatRot {
    quat: candle_core::Tensor,
}

impl NQuatRot {
    fn new(vb: &VarBuilder, n_splats: usize) -> Self {
        Self {
            quat: vb
                .get_with_hints(
                    &[n_splats, 4],
                    "quat",
                    candle_nn::init::DEFAULT_KAIMING_NORMAL,
                )
                .unwrap(),
        }
    }
    fn mat3(&self, device: &Device) -> candle_core::Result<candle_core::Tensor> {
        // let r = self.quat.i(0)?;
        // v_i = m_ijk Q^j Q^kj
        let two_s = (2.0 / (&self.quat * &self.quat)?.sum(1)?)?;
        let n = two_s.shape().dims()[0];
        let two_s = two_s.unsqueeze(1)?.expand(&[n, 9])?;
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
        let Mqq = M // 9 x 4 x 4
            .broadcast_matmul(
                &self
                    .quat // N x 4
                    .unsqueeze(2)? // N x 4 x 1
                    .unsqueeze(1)? // N x 9 x 4 x 1
                    .broadcast_as(&[n, 9, 4, 1])?,
            )?
            .t()?
            .broadcast_matmul(
                &self
                    .quat
                    .unsqueeze(2)? // N x 4 x 1
                    .unsqueeze(1)? // N x 9 x 4 x 1
                    .broadcast_as(&[n, 9, 4, 1])?,
            )?
            .squeeze(2)?
            .squeeze(2)?;
        // .broadcast_matmul(&self.quat.unsqueeze(1)?)?;
        println!("Mqq shape: {:?}", Mqq.shape());
        let two_s_M = (two_s * Mqq)?;
        let lhs =
            candle_core::Tensor::new(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], device);
        // let lhs = lhs?.unsqueeze(1)?.unsqueeze(1)?.expand(&[9, 4, 4])?;
        let x = (lhs - ((-1.0) * two_s_M)?)?;
        let x = x.reshape(&[3, 3])?;
        let proj = candle_core::Tensor::new(&[[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]], device)?;
        let x = proj.matmul(&x.matmul(&proj.t()?)?)?;
        println!("proj: {:?}", x.shape());
        // | 1 0 0 | | a b c |   | a b c |
        // | 0 1 0 | | d e f | = | d e f |
        //           | g h i |

        // | a b c | | 1 0 |   | a b |
        // | d e f | | 0 1 | = | d e |
        //           | 0 0 |
        // let x = x.matmul(&x.t()?)?;
        Ok(x)
    }
}
fn test_gauss(device: &Device) -> candle_core::Result<candle_core::Tensor> {
    // let mut screen = candle_core::Tensor::zeros(&[128, 128], DType::F32, device);
    let x = candle_core::Tensor::arange(0.0, 12.0f32, device)?;
    let y = candle_core::Tensor::arange(0.0, 12.0f32, device)?;
    let screen = candle_core::Tensor::meshgrid(&[&x, &y], true)?;
    let screen = candle_core::Tensor::stack(screen.as_slice(), 2)?;
    let mut pos = candle_core::Tensor::new(&[[0.0f32, 0.0], [12.0f32, 12.0]], device)?;
    // let mut cov = candle_core::Tensor::new(&[[[1.0, 0.0], [0.0, 1.0]], [[]]], device)?;
    pos = pos.unsqueeze(1)?.unsqueeze(1)?;
    let B = pos.shape().dims()[0];
    println!("dist: {:?}", pos.shape());
    // pos = pos.expand(&[0, 12, 12, 2])?;
    let dist =
        (screen.expand(&[B, 12, 12, 2])? - pos.broadcast_as(&[B, 12, 12, 2]))?.contiguous()?;
    // let dist = (&dist * &dist)?.sum(3)?;
    println!("dist: {}", dist);

    let varmap: VarMap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // let q = QuatRot::new(&vs);
    let cov = Cov2::new(&vs);
    let C = cov.cov(device)?;
    println!("cont: {}", C.is_contiguous());
    let d2 = dist.reshape(((), 2, 1))?;
    let c2 = C.unsqueeze(0)?.broadcast_as((d2.shape().dims()[0], 2, 2))?;
    let c3 = c2.broadcast_matmul(&d2)?;
    let c4 = (-1.0 * d2.t()?.matmul(&c3)?)?;
    let c5 = c4.exp()?;
    let c6 = c5.reshape((B, 12, 12))?;
    println!("{}", c6);
    // println!("cs: {}", cs.is_contiguous());
    // println!("cs: {:?}", cs.shape());
    // .matmul(&dist.unsqueeze(4)?.contiguous()?)?;
    // dist^i C_ij dist^j
    // let stacked = q.mat3(device)?;
    // println!("{}", stacked);
    Ok(x)
}
fn test2_gauss(device: &Device) -> candle_core::Result<candle_core::Tensor> {
    let varmap: VarMap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let x = candle_core::Tensor::arange(0.0, 12.0f32, device)?;
    let y = candle_core::Tensor::arange(0.0, 12.0f32, device)?;
    let screen = candle_core::Tensor::meshgrid(&[&x, &y], true)?;
    let screen = candle_core::Tensor::stack(screen.as_slice(), 2)?;

    let mut pos = candle_core::Tensor::new(&[[0.0f32, 0.0], [12.0f32, 12.0]], device)?;
    // pos = pos.unsqueeze(1)?.unsqueeze(1)?;

    let B = pos.shape().dims()[0];
    let dist =
        (screen.expand(&[B, 12, 12, 2])? - pos.broadcast_as(&[B, 12, 12, 2]))?.contiguous()?;

    let cov: Vec<_> = (0..2).map(|_| Cov2::new(&vs)).collect();
    let cov_mats: Vec<_> = cov.iter().map(|c| c.cov(device).unwrap()).collect();
    let covs = candle_core::Tensor::stack(&cov_mats.as_slice(), 0)?;
    println!("{:?}", covs.shape());
    // let C = cov.cov(device)?;
    // println!("cont: {}", C.is_contiguous());
    let d2 = dist.reshape(((), 2, 1))?;
    let c2 = covs
        .unsqueeze(1)?
        .broadcast_as((2, d2.shape().dims()[0], 2, 2))?;
    println!("c2: {:?}", c2.shape());

    // M_aij dx^ai dx^j

    // let c3 = c2.broadcast_matmul(&d2)?;
    // let c4 = (-1.0 * d2.t()?.matmul(&c3)?)?;
    // let c5 = c4.exp()?;
    // let c6 = c5.reshape((B, 12, 12))?;
    // println!("{}", c6);
    // println!("cs: {}", cs.is_contiguous());
    // .matmul(&dist.unsqueeze(4)?.contiguous()?)?;
    // dist^i C_ij dist^j
    // let stacked = q.mat3(device)?;
    // println!("{}", stacked);
    Ok(x)
}

fn test3_gauss(device: &Device) -> candle_core::Result<()> {
    let varmap: VarMap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let varmap2: VarMap = VarMap::new();
    let vs2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
    let splat = Splat::new(&vs);
    let splat2 = Splat::new(&vs2);
    // let splat3 = Splat::new(&vs2);
    let mut sgd = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        },
    )?;
    // splat.pos.slice_set(
    //     &candle_core::Tensor::new(&[6.0f32, 6.0], device).unwrap(),
    //     0,
    //     0,
    // )?;
    // splat.cov.rot.quat.slice_set(
    //     &candle_core::Tensor::new(&[1.0f32, 0.0, 0.0, 1.0], device).unwrap(),
    //     0,
    //     0,
    // )?;
    // splat.cov.scale.slice_set(
    //     &candle_core::Tensor::new(&[2.0f32, 1.0], device).unwrap(),
    //     0,
    //     0,
    // )?;
    let target = splat2.render(device, 12)?.detach();
    for step in 0..10000 {
        let screen = splat.render(device, 12)?;
        let loss = candle_nn::loss::mse(
            &screen, &target, // &candle_core::Tensor::ones((12, 12), DType::F32, device)?,
        )?;
        sgd.backward_step(&loss)?;
        println!("{}", loss.to_scalar::<f32>()?);
        // println!("scale: {}", splat.cov.scale);
        // println!("rot: {}", splat.cov.rot.quat);
        println!("pos: {}", splat.pos);
        println!("target: {}", splat2.pos);
        // println!("pos: {}", screen);
    }

    // let loss =
    // println!("{}", screen);
    Ok(())
}
fn main() -> candle_core::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    test3_gauss(&dev)?;
    Ok(())
}
