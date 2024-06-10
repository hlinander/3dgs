use std::collections::HashMap;
use std::process::exit;

// use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{linear, loss, Linear, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use itertools::*;
use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::{Result, Write};

mod gauss;
mod keeloq;

struct KeeLoq {
    key: candle_core::Tensor,
    nlf: candle_core::Tensor,
    // static KeeLoq_NLF: u32 = 0x3A5C742E;
}

impl KeeLoq {
    fn new(vs: &VarBuilder, device: &Device, key_val: u64) -> Self {
        // candle_core::Var
        let key = vs
            .get_with_hints(&[64], "key", candle_nn::init::DEFAULT_KAIMING_NORMAL)
            .unwrap();
        key.slice_set(
            &candle_core::Tensor::from_vec(u64_to_vec_of_bits_as_f32(key_val), &[64], device)
                .unwrap(),
            0,
            0,
        )
        .unwrap();
        let mut nlf_vec = u32_to_vec_of_bits_as_f32(0x3A5C742Eu32);
        nlf_vec.reverse();
        Self {
            key,
            nlf: candle_core::Tensor::from_vec(nlf_vec, &[32], device).unwrap(),
        }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = xs.clone();
        // println!("key: {}", self.key);
        for r in 0..528 {
            let x_shift = x.i((.., ..31))?;
            let b0 = x.i((.., 31 - 0))?;
            let b16 = x.i((.., 31 - 16))?;
            let kbr = candle_nn::ops::sigmoid(&self.key.i(63 - (r % 64)).unwrap())?;
            // let kbr = &self.key.i(63 - (r % 64))?;
            // println!("kbr at [{}]: {}", r, kbr);
            let xor = (b0 - b16).unwrap().abs().unwrap().unsqueeze(1).unwrap();
            // println!("b016 at [{}]: {}", r, xor);
            // if r > 20 {
            // return Ok(x);
            // }
            let shape = xor.shape().clone();
            // println!("{} ^ {}", xor, kbr);
            let xor = (xor - kbr.unsqueeze(0).unwrap().broadcast_as(shape))
                .unwrap()
                .abs()
                .unwrap();

            let b1 = x.i((.., 31 - 1))?;
            let b9 = x.i((.., 31 - 9))?;
            let b20 = x.i((.., 31 - 20))?;
            let b26 = x.i((.., 31 - 26))?;
            let b31 = x.i((.., 31 - 31))?;
            let g5 =
                (((b1 + b9 * 2.0).unwrap() + b20 * 4.0).unwrap() + b26 * 8.0).unwrap() + b31 * 16.0;
            let g5 = g5.unwrap().clamp(0.0, 31.9);
            let g5 = g5.unwrap().floor().unwrap().to_dtype(DType::U32).unwrap();
            // println!("g5 at [{}]: {}", r, g5);
            // println!("nlf: {}", self.nlf);
            let nlf_bit = self
                .nlf
                .unsqueeze(1)
                // .broadcast_as(&[32, 1])
                .unwrap()
                .embedding(&g5)
                .unwrap();
            // println!("nlf_bit at [{}]: {}", r, nlf_bit);
            // let shape = xor.shape().clone();
            // println!("{:?}", shape);
            let xor = (xor - nlf_bit).unwrap().abs().unwrap();
            // println!("xor at [{}] {}", r, xor);
            // let xor = (xor - nlf_bit)
            // self.nlf.embedding()
            // let xor = (xor - kbr.expand(xor.shape())).unwrap().abs().unwrap();
            // println!("{:?}", xor.shape());
            // println!("{:?}", x_shift.shape());
            x = candle_core::Tensor::cat(&[&xor, &x_shift], 1)?;
            // let out_bits = x.to_vec2::<f32>().unwrap();
            // println!(
            // "output bits at [{}]: {:x}",
            // r,
            // vec_of_floats_to_u32(out_bits[0].clone())
            // );

            // println!("{:?}", x.shape());
            // x = (x_shift - b0)
        }
        Ok(x)
    }
}

struct BitModel {
    layers: Vec<Linear>,
    // basis_boys: HashMap<Vec<usize>, usize>, // basis boy to index
}

impl BitModel {
    fn new(
        vs: &VarBuilder,
        in_dim: usize,
        out_dim: usize,
        // n_points: usize,
        widths: Vec<usize>,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let interior_layers = widths
            .iter()
            .tuple_windows()
            .enumerate()
            .map(|(idx, (a, b))| linear(*a, *b, vs.pp(format!("l{}", idx))).unwrap());
        let in_layer = linear(in_dim, widths[0], vs.pp("in_layer"))?;
        let out_layer = linear(widths[widths.len() - 1], out_dim, vs.pp("out_layer"))?;
        // let out_layer = (vs, "out", widths[widths.len() - 1], n_points, device)
        // .context("out layer new")?;
        Ok(Self {
            layers: std::iter::once(in_layer)
                .chain(interior_layers)
                .chain([out_layer])
                .collect(),
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut iter = xs.clone();
        for (idx, layer) in self.layers.iter().take(self.layers.len() - 1).enumerate() {
            iter = layer.forward(&iter)?;
            iter = iter.relu()?;
            // candle_nn::ops::leaky_relu(, )
            // candle_core::r
            // iter = candle_nn::activation::prelu(, )
            // .with_context(|| format!("layer {}", idx))?;
        }
        iter = self.layers[self.layers.len() - 1].forward(&iter)?;
        Ok(iter)
        // Ok(self.extract_clifford(&iter)?)
    }
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
}

fn u32_to_vec_of_bits_as_f32(value: u32) -> Vec<f32> {
    let mut bits = Vec::with_capacity(32);
    for i in (0..32).rev() {
        bits.push(if value & (1 << i) != 0 { 1.0 } else { 0.0 });
    }
    bits
}

fn u64_to_vec_of_bits_as_f32(value: u64) -> Vec<f32> {
    let mut bits = Vec::with_capacity(64);
    for i in (0..64).rev() {
        bits.push(if value & (1 << i) != 0 { 1.0 } else { 0.0 });
    }
    bits
}

fn vec_of_floats_to_u32(bits: Vec<f32>) -> u32 {
    let mut result: u32 = 0;
    for (index, &value) in bits.iter().enumerate() {
        if value > 0.5 {
            result |= 1 << (bits.len() - 1 - index);
        }
    }
    result
}

type FloatBits = Vec<f32>;

struct Batch {
    // Many keys
    keys: Vec<u64>,

    // Bits as Vec<f32> for every key
    key_bits: Vec<f32>,

    // For every key, there are multiple payloads
    // these are concatenated to one Vec<u32>
    payloads: Vec<u32>,

    // Corresponding bits for each payload, again
    // concatenated.
    payload_bits: Vec<f32>,

    // Encrypted payloads, concatenated
    encrypted: Vec<u32>,

    // Bits of encrypted payloads, concatenated
    encrypted_bits: Vec<f32>,
}

fn training_loop(args: &TrainingArgs) -> candle_core::Result<()> {
    let dev = candle_core::Device::cuda_if_available(0)?;
    println!("{}", dev.is_cuda());

    let varmap: VarMap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let payloads_per_key = 1000usize;
    let n_keys_per_batch = 10usize;
    // let model = BitModel::new(&vs, 32 * payloads_per_key, 64, vec![256, 256], &dev)?;
    let thekey = 0x5cec6701b79fd949;
    // let keeloq = KeeLoq::new(&vs, &dev, thekey);
    let keeloq = KeeLoq::new(&vs, &dev, 0x1234);
    let key_bits = u64_to_vec_of_bits_as_f32(thekey);
    let key_bits_tensor = candle_core::Tensor::from_vec(key_bits, &[64], &dev).unwrap();
    // let keeloq = KeeLoq::new(&vs, &dev, 0x31231231);
    // keeloq.forward()
    println!("one_in_bits: {:?}", u32_to_vec_of_bits_as_f32(0x1));
    println!("key bits: {:?}", u64_to_vec_of_bits_as_f32(thekey));
    let test_in =
        candle_core::Tensor::from_vec(u32_to_vec_of_bits_as_f32(0xf741e2db), &[1, 32], &dev)
            .unwrap();
    let test = keeloq.forward(&test_in).unwrap();
    // test.backward();
    let gt = keeloq::KeeLoq_Encrypt(0xf741e2db, thekey);
    println!("output: {}", test);
    let out_bits = test.to_vec2::<f32>().unwrap();
    println!("{:?}", out_bits);
    println!("gt: {:x}", gt);
    println!(
        "output bits: {:x}",
        vec_of_floats_to_u32(out_bits[0].clone())
    );
    let t = vec_of_floats_to_u32(u32_to_vec_of_bits_as_f32(0xf741e2db));
    println!("{:x}", t);
    // exit(0);
    // let model = BitModel::new(&vs, 64, 64, vec![256, 256], &dev)?;

    let mut sgd = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        },
    )?;
    // let mut sgd = candle_nn::SGD::new(varmap.all_vars(), args.learning_rate)?;

    let mut rng32 = rand::thread_rng();
    let mut rng64 = rand::thread_rng();
    let mut loss_file = File::create("loss.txt").unwrap();
    for epoch in 1..args.epochs {
        let payloads = (0..1000)
            .map(|_| rng32.gen())
            // .flat_map(|x| x.into_iter())
            .collect_vec();
        let payload_bits = payloads
            .clone()
            .into_iter()
            .map(|x| u32_to_vec_of_bits_as_f32(x))
            .flat_map(|x| x.into_iter())
            .collect_vec();
        let encrypted_payloads = payloads
            .clone()
            .into_iter()
            .map(|payload| u32_to_vec_of_bits_as_f32(keeloq::KeeLoq_Encrypt(payload, thekey)))
            .flat_map(|x| x.into_iter())
            .collect_vec();
        let in_tensor = candle_core::Tensor::from_vec(payload_bits, &[1000, 32], &dev).unwrap();
        let target_tensor =
            candle_core::Tensor::from_vec(encrypted_payloads, &[1000, 32], &dev).unwrap();
        // println!("before model");
        let res = keeloq.forward(&in_tensor).unwrap();
        // println!("input: {}", res);
        // println!("target: {}", target_tensor);
        // let loss = loss::binary_cross_entropy_with_logit(&res, &target_tensor)?;
        let loss = loss::mse(&res, &target_tensor).unwrap();
        println!("loss {}", loss);
        // exit(0);
        sgd.backward_step(&loss)?;
        writeln!(loss_file, "{}", loss.to_scalar::<f32>().unwrap()).unwrap();
        loss_file.flush()?;
        // println!("{}", keeloq.key);
        // println!("{}", loss.to_scalar::<f32>().unwrap());
    }
    for epoch in 1..args.epochs {
        let batch = generate_batch(payloads_per_key, n_keys_per_batch, &mut rng32, &mut rng64);
        let mut intensor = candle_core::Tensor::from_vec(
            batch.encrypted_bits,
            candle_core::Shape::from_dims(&[n_keys_per_batch, payloads_per_key * 32]),
            &dev,
        )
        .unwrap();
        intensor = (intensor - 0.5).unwrap();
        let target_tensor = candle_core::Tensor::from_vec(
            batch.key_bits.clone(),
            candle_core::Shape::from_dims(&[n_keys_per_batch, 64usize]),
            &dev,
        )
        .unwrap();
        // let mut in_target_tensor = candle_core::Tensor::from_vec(
        //     batch.key_bits,
        //     candle_core::Shape::from_dims(&[n_keys_per_batch, 64usize]),
        //     &dev,
        // )
        // .unwrap();
        // in_target_tensor = (in_target_tensor - 0.5).unwrap();
        // let intensor = candle_core::Tensor::ones((3, 66), DType::F32, &dev)?;
        // let target_tensor = candle_core::Tensor::ones((3, 16), DType::F32, &dev)?;
        // println!("{}", intensor);
        // let logits = model.forward(&in_target_tensor)?;
        let logits = keeloq.forward(&intensor)?;
        let loss = loss::binary_cross_entropy_with_logit(&logits, &target_tensor)?;
        println!("{loss}");
        writeln!(loss_file, "{}", loss.to_scalar::<f32>().unwrap()).unwrap();
        loss_file.flush()?;
        // loss_file.write()
        sgd.backward_step(&loss)?;
    }
    Ok(())
}

fn generate_batch(
    payloads_per_key: usize,
    n_keys: usize,
    rng32: &mut rand::rngs::ThreadRng,
    rng64: &mut rand::rngs::ThreadRng,
) -> Batch {
    let mut keys = Vec::with_capacity(n_keys);
    let mut key_bits: Vec<f32> = Vec::with_capacity(n_keys * 64);
    let mut all_payloads: Vec<u32> = Vec::with_capacity(payloads_per_key * n_keys);
    let mut all_payload_bits: Vec<f32> = Vec::with_capacity(payloads_per_key * n_keys * 32);
    let mut encrypted: Vec<u32> = Vec::with_capacity(payloads_per_key * n_keys);
    let mut encrypted_bits: Vec<f32> = Vec::with_capacity(payloads_per_key * n_keys * 32);
    for key_idx in 0..n_keys {
        let index: u32 = rng32.gen();
        let key: u64 = [
            0x89893213,
            0x9449324,
            0xa8a898a9a,
            0xb898a,
            0x12345u64,
            0x4893724u64,
            0x9213123u64,
        ][(index % 7) as usize];
        let key: u64 = rng64.gen();
        // let payloads: Vec<u32> = (0..payloads_per_key).map(|_| rng32.gen()).collect_vec();
        let payloads: Vec<u32> = (0..payloads_per_key).map(|_| 0xABCDEFu32).collect_vec();
        let encrypted_payloads = payloads
            .clone()
            .into_iter()
            .map(|payload| keeloq::KeeLoq_Encrypt(payload, key))
            .collect_vec();
        encrypted.extend(encrypted_payloads.clone());
        let encrypted_payloads_bits = encrypted_payloads
            .iter()
            .map(|encrypted| u32_to_vec_of_bits_as_f32(*encrypted))
            .flat_map(|encrypted| encrypted.into_iter())
            .collect_vec();
        assert_eq!(encrypted_payloads_bits.len(), payloads_per_key * 32);
        encrypted_bits.extend(encrypted_payloads_bits);
        all_payload_bits.extend(
            payloads
                .clone()
                .into_iter()
                .flat_map(|payload| u32_to_vec_of_bits_as_f32(payload)),
        );
        all_payloads.extend(payloads);
        key_bits.extend(u64_to_vec_of_bits_as_f32(key));
        keys.push(key);
    }
    assert_eq!(encrypted_bits.len(), payloads_per_key * n_keys * 32);
    Batch {
        keys,
        key_bits,
        payloads: all_payloads,
        payload_bits: all_payload_bits,
        encrypted,
        encrypted_bits,
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 2000000)]
    epochs: usize,
}

pub fn main() -> candle_core::Result<()> {
    let args = Args::parse();
    // Load the dataset
    let default_learning_rate = 0.001;
    let training_args = TrainingArgs {
        epochs: args.epochs,
        learning_rate: args.learning_rate.unwrap_or(default_learning_rate),
    };
    training_loop(&training_args)
}
