static KeeLoq_NLF: u32 = 0x3A5C742E;

fn bit(x: u32, n: u32) -> u32 {
    ((x) >> (n)) & 1
}

fn bit64(x: u64, n: u32) -> u64 {
    ((x) >> (n)) & 1
}

fn g5(x: u32, a: u32, b: u32, c: u32, d: u32, e: u32) -> u32 {
    bit(x, a) + bit(x, b) * 2 + bit(x, c) * 4 + bit(x, d) * 8 + bit(x, e) * 16
}

pub fn KeeLoq_Encrypt(data: u32, key: u64) -> u32 {
    let mut x: u32 = data;
    for r in 0..528 {
        // let b016 = bit(x, 0) ^ bit(x, 16);
        // let kbr = bit64(key, r & 63);
        // let g5val = g5(x, 1, 9, 20, 26, 31);
        // let kbit = bit(KeeLoq_NLF, g5val);
        // let xor_val = (bit(x, 0)
        //     ^ bit(x, 16)
        //     ^ (bit64(key, r & 63) as u32)
        //     ^ bit(KeeLoq_NLF, g5(x, 1, 9, 20, 26, 31)));
        // println!("b016 at [{}]: {}", r, b016);
        // println!("kbr at [{}]: {}", r, kbr);
        // println!("g5 at [{}]: {}", r, g5val);
        // println!("kbit at [{}]: {}", r, kbit);
        // println!("xor_val at [{}]: {}", r, xor_val);

        // if r > 20 {
        // return 0;
        // }
        x = (x >> 1)
            ^ ((bit(x, 0)
                ^ bit(x, 16)
                ^ (bit64(key, r & 63) as u32)
                ^ bit(KeeLoq_NLF, g5(x, 1, 9, 20, 26, 31)))
                << 31);
        // println!("x_out at [{}]: {:x}", r, x);
    }
    return x;
}

pub fn KeeLoq_Decrypt(data: u32, key: u64) -> u32 {
    let mut x: u32 = data;

    for r in 0..528 {
        x = (x << 1)
            ^ bit(x, 31)
            ^ bit(x, 15)
            ^ (bit64(key, 15u32.overflowing_sub(r).0 & 63u32) as u32)
            ^ bit(KeeLoq_NLF, g5(x, 0, 8, 19, 25, 30));
    }

    return x;
}
use std::time::Instant;

fn main() {
    // key                | plaintext  | ciphertext
    // 0x5cec6701b79fd949 | 0xf741e2db | 0xe44f4cdf
    // 0x5cec6701b79fd949 | 0x0ca69b92 | 0xa6ac0ea2

    // let key = 0x5cec6701b79fd949u64;

    // println!("cipher: 0x{:x}", KeeLoq_Encrypt(0xf741e2dbu32, key));
    // println!("cipher: 0x{:x}", KeeLoq_Encrypt(0x0ca69b92, key));

    // println!(
    //     "identity: 0x{:x}",
    //     KeeLoq_Decrypt(KeeLoq_Encrypt(0xf741e2dbu32, key), key)
    // );
    // println!(
    //     "identity: 0x{:x}",
    //     KeeLoq_Decrypt(KeeLoq_Encrypt(0x0ca69b92, key), key)
    // );
    // let start = Instant::now();

    for key in 0..0xFFFFF {
        let val = KeeLoq_Decrypt(0, key);
        if key % 0xFFFFF == 0 {
            println!("0x{:x}", val);
        }
    }

    // let duration = start.elapsed();
    // println!("Time elapsed: {:?}", duration);
    // println!(
    //     "Time elapsed: {:?}",
    //     duration.as_secs() * (u64::MAX / 0xFFFFFFu64)
    // );
}
