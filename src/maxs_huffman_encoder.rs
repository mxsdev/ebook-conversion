use std::{
    collections::{hash_map::Entry, HashMap},
    io::Cursor,
    ops::RangeInclusive,
    str::from_utf8,
};

use bitvec::{
    order::{BitOrder, Lsb0, Msb0},
    view::{AsBits, BitView},
};

use byteorder::{BigEndian, ReadBytesExt};
use itertools::Itertools;
use rand::Rng;

use crate::huffman::HuffmanTable;

pub struct MaxsHuffmanEncoder;

type CodeDict = [Option<(u8, bool, u32)>; 256];

struct EncodedResult {
    chunks: Vec<Chunk>,
    code_dict: CodeDict,
}

impl Default for EncodedResult {
    fn default() -> Self {
        Self {
            chunks: Default::default(),
            code_dict: [None; 256],
        }
    }
}

fn advance_stk(stk: &mut Vec<Chunk>, prefix: u32, bits: u32) -> bool {
    if stk.len() == 0 {
        return true;
    }

    let mut res = true;

    let mut last_chunk_idx = 0;
    for chunk in stk.iter_mut() {
        if !chunk.advance_if_valid(prefix, bits) {
            res = false;
            break;
        }

        last_chunk_idx += 1;
    }

    if last_chunk_idx != stk.len() {
        for (i, chunk) in stk.iter_mut().enumerate() {
            if i >= last_chunk_idx {
                break;
            }

            chunk.backtrack(bits)
        }
    }

    res
}

impl MaxsHuffmanEncoder {
    fn rec_helper(
        data: &[u8],
        i: usize,
        last_dcis: (Option<u32>, Option<u32>),
        codes: &FreqMap,
        code_dict: &mut CodeDict,
        stk: &mut Vec<Chunk>,
        resp: &mut EncodedResult,
    ) -> bool {
        // if let Some(last_dci) = last_dci {
        //     let (last_bi, ..) = code_dict[last_dci as usize].unwrap();

        //     let new_bits = last_dci & ((1 << (8 - last_bi)) - 1);

        //     let last = stk.pop().unwrap();

        //     if !advance_stk(stk, new_bits, 8 - last_bi as u32) {
        //         stk.push(last);
        //         return false;
        //     }

        //     stk.push(last);

        //     for chunk in stk.iter() {
        //         println!(
        //             "prefix: {:#34b}, pushed: {:?}, pushed dci: {:?}",
        //             chunk.prefix, last_bi, last_dci
        //         )
        //     }
        // }

        if i >= data.len() {
            // complete stk
            let mut valid_range = RangeInclusive::<u32>::new(u32::MIN, u32::MAX);
            let largest_rem_bits = 24 - stk.last().unwrap().i;

            if largest_rem_bits > 0 {
                for chunk in stk.iter() {
                    println!(
                        "finalizing chunk, i: {:?}, prefix: {:#34b}",
                        chunk.i, chunk.prefix
                    );

                    let rem_bits = 24_u32.saturating_sub(chunk.i);
                    if rem_bits <= 0 {
                        continue;
                    }

                    let lhs =
                        (chunk.lhs() & ((1 << rem_bits) - 1)) << (largest_rem_bits - rem_bits);
                    let rhs = ((chunk.rhs() & ((1 << rem_bits) - 1))
                        << (largest_rem_bits - rem_bits))
                        + ((1 << largest_rem_bits - rem_bits) - 1);

                    valid_range = RangeInclusive::<u32>::new(
                        (lhs).max(*valid_range.start()),
                        (rhs).min(*valid_range.end()),
                    );
                }

                if valid_range.is_empty() {
                    return false;
                }

                let rem = valid_range.into_iter().next().unwrap();

                println!("rem: {:#34b}", rem);

                for chunk in stk.iter_mut() {
                    let rem_bits = 24_u32.saturating_sub(chunk.i);
                    if rem_bits <= 0 {
                        continue;
                    }

                    if !chunk.advance_if_valid(rem, largest_rem_bits) {
                        panic!("unexpected...")
                    }
                }

                let mut res = true;

                for chunk in stk.iter_mut() {
                    // TODO: assert this
                    if chunk.lhs_ignoring_prefix() > chunk.prefix {
                        res = false;
                    }

                    // TODO: assert this
                    if chunk.rhs_ignoring_prefix() < chunk.prefix {
                        res = false;
                    }

                    let (_, _, X) = code_dict[chunk.dci as usize].unwrap();
                    if chunk.prefix > X {
                        res = false;
                    }

                    // let mut last_res: Option<u32> = None;
                    // let (_, _, X) = code_dict[chunk.dci as usize].unwrap();
                    // for x in chunk.iter_ignoring_prefix() {
                    //     let res = (X - ((chunk.dci << 24) + x)) >> (32 - chunk.bi);
                    //     if let Some(last_res) = last_res {
                    //         assert_eq!(res, last_res);
                    //     }
                    //     last_res = Some(res);
                    // }
                    // let res = (X - ((chunk.dci << 24) + chunk.prefix)) >> (32 - chunk.bi);
                    // println!(
                    //     "chunk code: {:?}, G: {:?}, prev: {:?}, lhs: {:?}, rhs: {:?}, prefix: {:?}, is_valid: {:?}",
                    //     res,
                    //     chunk.G,
                    //     last_res,
                    //     chunk.lhs_ignoring_prefix(),
                    //     chunk.rhs_ignoring_prefix(),
                    //     chunk.prefix,
                    //     chunk.check_valid(),
                    // );

                    // chunk dci: 0, bi: 4, value: 0b00000000000011111010000000010001
                    // chunk dci: 0, bi: 4, value: 0b    00000000111110100000000100010000

                    // chunk dci: 0, bi: 4, value: 0b00000000000010101100111110001100
                    // chunk dci: 0, bi: 4, value: 0b    00000000101011001111100011000000

                    // assert!(chunk.lhs_ignoring_prefix() <= chunk.prefix);
                    // assert!(chunk.rhs_ignoring_prefix() >= chunk.prefix);
                    // assert!(chunk.i >= 24);
                }

                if !res {
                    for chunk in stk.iter_mut() {
                        let rem_bits = 24_u32.saturating_sub(chunk.i);
                        if rem_bits <= 0 {
                            continue;
                        }

                        chunk.backtrack(largest_rem_bits);
                    }
                }
            }

            // println!("code_dict: {:?}", code_dict);

            resp.chunks = (*stk).clone();
            resp.code_dict = (*code_dict).clone();
            return true;
        }

        let char = data[i];

        let cci = codes.get_code(char);
        let char_code_len = codes.get_code_len(char);

        for dci in 0..=255 {
            // println!("i: {:?}, try dci {:?}...", i, dci);

            let _ = if let Some(last_dci) = last_dcis.0 {
                let (last_code_len, ..) = code_dict[last_dci as usize].unwrap();

                if dci >> last_code_len != ((1 << (8 - last_code_len)) - 1) & last_dci {
                    continue;
                }

                (((1 << (8 - last_code_len)) - 1) & dci, 8 - last_code_len)
            } else {
                (dci, 8)
            };

            let code_pushed = (dci as u32) << 24;
            if code_dict[dci as usize].is_none() {
                for bi in char_code_len.max(4)..=8 {
                    let mut is_valid = true;
                    for chunk in stk.iter_mut() {
                        if !chunk.can_advance(dci >> (8 - bi as u32), bi as u32) {
                            is_valid = false;
                        }
                    }

                    if !is_valid {
                        continue;
                    }

                    let C = code_pushed.wrapping_add((cci as u32) << (32 - bi));

                    let max_Z: u32 = (1 << 32 - bi) - 1;
                    let max_Y: u32 = (1 << 24) - 1;

                    for _ in 0..100 {
                        let range = max_Z..(max_Z + max_Y - 1);

                        let G = rand::thread_rng().gen_range(range);

                        if G.overflowing_add(C).1 {
                            continue;
                        }

                        let X = G + C;

                        if code_pushed > X {
                            continue;
                        }

                        let chunk = Chunk::new(bi, G, dci);

                        if !chunk.check_valid() {
                            return false;
                        }

                        // for x in chunk.iter_igonring_prefix() {
                        //     assert_eq!((X - ((dci << 24) + x)) >> (32 - bi), cci as u32);
                        // }

                        // println!("lhs: {:?}, rhs: {:?}", chunk.lhs(), chunk.rhs());

                        stk.push(chunk);
                        code_dict[dci as usize] = (bi, true, X).into();

                        println!("dci: {:?}, bi: {:?}, X: {:?}", dci, bi, X);
                        if Self::rec(
                            data,
                            i + 1,
                            (dci.into(), last_dcis.0),
                            codes,
                            code_dict,
                            stk,
                            resp,
                        ) {
                            return true;
                        }

                        code_dict[dci as usize] = None;
                        stk.pop();
                    }
                }
            } else {
                let (bi, _, X) = code_dict[dci as usize].unwrap();

                let max_Z: u32 = (1 << 32 - bi) - 1;

                let C = code_pushed.wrapping_add((cci as u32) << (32 - bi));

                if X < C {
                    return false;
                }

                if code_pushed > X {
                    continue;
                }

                let G = X - C;

                if G < max_Z {
                    return false;
                }

                let mut chunk = Chunk::new(bi, G, dci);
                if !chunk.check_valid() {
                    return false;
                }
                // if !chunk.advance_if_valid(dci >> (8 - bi), bi as u32) {
                //     // panic!("should never fail")
                //     continue;
                // }

                stk.push(chunk);
                if Self::rec(
                    data,
                    i + 1,
                    (dci.into(), last_dcis.0),
                    codes,
                    code_dict,
                    stk,
                    resp,
                ) {
                    return true;
                }
                stk.pop();
            }
        }

        // if let Some(last_dci) = last_dci {
        //     let (last_bi, ..) = code_dict[last_dci as usize].unwrap();

        //     for chunk in stk.iter_mut() {
        //         chunk.backtrack(last_bi as u32);
        //     }
        // }

        false
    }

    pub fn rec(
        data: &[u8],
        i: usize,
        last_dcis: (Option<u32>, Option<u32>),
        codes: &FreqMap,
        code_dict: &mut CodeDict,
        stk: &mut Vec<Chunk>,
        resp: &mut EncodedResult,
    ) -> bool {
        if let Some((last_dci, _)) = last_dcis.0.zip(last_dcis.1) {
            let (last_bi, ..) = code_dict[last_dci as usize].unwrap();

            let new_bits = last_dci & ((1 << (8 - last_bi)) - 1);

            let last = stk.pop().unwrap();

            if !advance_stk(stk, new_bits, 8 - last_bi as u32) {
                stk.push(last);
                return false;
            }

            stk.push(last);

            // for chunk in stk.iter() {
            //     println!(
            //         "prefix: {:#34b}, pushed: {:?}, pushed dci: {:?}, i: {:?}",
            //         chunk.prefix,
            //         8 - last_bi,
            //         last_dci,
            //         chunk.i
            //     )
            // }
        }

        let res = Self::rec_helper(data, i, last_dcis, codes, code_dict, stk, resp);

        if res {
            return res;
        }

        if let Some((last_dci, _)) = last_dcis.0.zip(last_dcis.1) {
            let (last_bi, ..) = code_dict[last_dci as usize].unwrap();

            let last = stk.pop().unwrap();
            for chunk in stk.iter_mut() {
                chunk.backtrack(8 - last_bi as u32);
            }
            stk.push(last);
        }

        res
    }

    pub fn pack(&mut self, data: &[u8]) -> (HuffmanTable, Vec<u8>) {
        let freq_map = freq_map(data);

        let mut code_dict = [None; 256];
        let mut stk = Default::default();
        let mut resp = Default::default();

        if !Self::rec(
            data,
            0,
            (None, None),
            &freq_map,
            &mut code_dict,
            &mut stk,
            &mut resp,
        ) {
            panic!("failed to recurse...")
        }

        let mut table = HuffmanTable {
            dictionary: freq_map
                .code_to_char
                .iter()
                .map(|char| Some((Vec::from([*char]), true)))
                .collect(),
            code_dict: resp.code_dict.map(|val| match val {
                Some(x) => x,
                None => (0, false, 0),
            }),
            ..Default::default()
        };

        let mut stk = resp.chunks;

        for chunk in stk.iter() {
            // println!("chunk prefix: {:#034b}", chunk.prefix);
            println!(
                "chunk dci: {:?}, bi: {:?}, value: {:#034b}",
                chunk.dci,
                chunk.bi,
                chunk.prefix + (chunk.dci << 24)
            );

            // 0b 00000000000011101111000110100010
            // 0b     00000001  111000011101000011010000

            // println!(
            //     "chunk bits: {:?}",
            //     (chunk.dci as u8)
            //         .view_bits::<Msb0>()
            //         .into_iter()
            //         .map(|b| *b)
            //         .collect::<Vec<_>>()
            // )
            // println!("chunk: {:#034b}", chunk.prefix + (chunk.dci << 24));
        }

        let last = stk.pop().unwrap();

        let mut bit_iterator = stk
            .into_iter()
            .flat_map(|chunk| {
                let bi = chunk.bi as usize;
                (chunk.dci as u8).view_bits::<Msb0>()[0..bi]
                    .into_iter()
                    .map(|b| *b)
                    .collect::<Vec<_>>()
            })
            .chain(
                (last.prefix + (last.dci << 24))
                    .view_bits::<Msb0>()
                    .into_iter()
                    .map(|b| *b)
                    .collect::<Vec<_>>()
                    .into_iter(),
            )
            .peekable();

        let mut bytes: Vec<u8> = Default::default();

        while bit_iterator.peek().is_some() {
            let mut byte: u8 = 0;
            for _ in 0..8 {
                byte <<= 1;
                let n = bit_iterator.next();

                byte += n.and_then(|x| x.then_some(1)).unwrap_or_default();
            }

            println!("output byte: {:#010b}", byte);

            bytes.push(byte);
        }

        // println!("{:?}", bytes);

        let mut decoded = Vec::<u8>::default();

        // simple decoder
        // let mut r = BigEndianReader::new(bytes.as_slice());
        let bits = bytes.as_bits::<Msb0>();
        let mut i = 0;

        // println!("{:?}, {:?}", bytes.len(), r.position());

        let mut decoded_length = data.len();

        while bits.len() - i >= 32 && decoded.len() < decoded_length {
            let mut x: u32 = 0;
            for j in 0..32 {
                x <<= 1;
                x += if bits[i + j] { 1 } else { 0 };
            }

            let dci = x >> 24;
            let (bi, _, X) = table.code_dict[dci as usize];
            println!("decoding, dci: {:?}, bi: {:?}, X: {:?}", dci, bi, X);

            i += bi as usize;

            let code = (X - x) >> (32 - bi);
            let char = table.dictionary[code as usize].as_ref().unwrap().0[0];

            decoded.push(char);
        }

        println!("{:?}", from_utf8(&decoded).unwrap());

        (table, bytes)

        // let res = stk.into_iter().map(|s| {
        //     s.prefix
        // })

        // let mut num_bits = 0;
        // for chunk in stk {
        //     chunk.prefix
        // }

        // let res: Vec<u8> = Default::default();
        // let mut bit_idx: usize = 0;

        // let mut code_dict = CodeDict::from([None; 256]);

        // let mut next_code_len = 4;
        // let mut code_idx = 0;

        // let mut res = Vec::<u32>::new();

        // let mut dci: u8 = 0;

        // for char in data {
        //     let cci = codes.get_code(*char);
        //     let char_code_len = codes.get_code_len(*char);

        //     // try to minimize
        //     let bi = char_code_len;

        //     // code_dict[code as usize] = Some((char_code_len, true, X));
        //     let code_pushed = (dci as u32) << 24;

        //     let C = code_pushed.wrapping_add((cci as u32) << (32 - bi));

        //     let mut Y = 0u32;

        //     let max_Z: u32 = (1 << 32 - bi) - 1;
        //     let max_Y: u32 = (1 << 24) - 1;

        //     // choose G > max_Z
        //     for G in max_Z..(max_Z + max_Y) {
        //         let X = G.wrapping_sub(C);
        //         for Y in G - max_Z..=max_Y.min(G) {
        //             let next_value = code_pushed + Y;

        //             println!(
        //                 "{:?}, {:?}, {:?}",
        //                 codes.get_char(((X - next_value) >> (32 - bi)) as u8),
        //                 next_value >> 24,
        //                 u32_size(Y),
        //             );
        //         }
        //     }

        //     // TODO: choose G <= max_Z

        //     // if G > max_Z {
        //     //     for Y in G - max_Z..=max_Y.min(G) {
        //     //         let next_value = code_pushed + Y;

        //     //         println!(
        //     //             "{:?}",
        //     //             freq.get_char(((X - next_value) >> (32 - char_code_len)) as u8)
        //     //         );
        //     //     }
        //     // }

        //     // }

        //     // res.push(next_value);

        //     // println!("{:?}", u32_size(Y));
        //     // println!("flags: {:#034b}", Y);

        //     // for Y in G.wrapping_sub(1 << (32 - char_code_len)).wrapping_add(1)..1 << 24 {
        //     //     println!("size: {:#034b}", Y);
        //     // }

        //     // for Y in 0..G.min(1 << 24) {
        //     //     // println!("size: {:#034b}", Y);

        //     //     let next_value = code_pushed + Y;

        //     //     println!(
        //     //         "{:?}",
        //     //         freq.get_char(((X - next_value) >> (32 - char_code_len)) as u8)
        //     //     );
        //     // }

        //     code_idx += 1;
        // }

        // println!("{:?}", code_pushed);

        // (Default::default(), &[])
    }
}

#[derive(Debug, Copy, Clone)]
struct Chunk {
    bi: u8,
    G: u32,
    dci: u32,

    prefix: u32,
    i: u32,
}

impl Chunk {
    pub fn new(bi: u8, G: u32, dci: u32) -> Self {
        let max_Z = (1 << (32 - bi)) - 1;

        if G < max_Z {
            panic!("G must be >= max_Z")
        }

        Self {
            bi,
            G,
            dci,
            i: 0,
            prefix: 0,
        }
    }

    fn iter(&self) -> impl Iterator<Item = u32> {
        self.lhs()..=self.rhs()
    }

    fn iter_ignoring_prefix(&self) -> impl Iterator<Item = u32> {
        self.lhs_ignoring_prefix()..=self.rhs_ignoring_prefix()
    }

    fn lhs_ignoring_prefix(&self) -> u32 {
        let max_Z = (1 << (32 - self.bi)) - 1;

        if self.G < max_Z {
            panic!("G must be >= max_Z")
        }

        self.G - max_Z
    }

    fn lhs(&self) -> u32 {
        if self.i >= 24 {
            return self.prefix;
        }

        self.lhs_ignoring_prefix().max(self.prefix << (24 - self.i))
    }

    fn rhs_ignoring_prefix(&self) -> u32 {
        let max_Y = (1 << 24) - 1;
        max_Y
    }

    fn rhs(&self) -> u32 {
        if self.i >= 24 {
            return self.prefix;
        }

        self.rhs_ignoring_prefix()
            .min((self.prefix << (24 - self.i)) + ((1 << (24 - self.i)) - 1))
    }

    fn check_in_range(&self, val: u32) -> bool {
        val >= self.rhs() && val <= self.lhs()
    }

    fn check_valid(&self) -> bool {
        self.rhs() >= self.lhs()
    }

    fn backtrack(&mut self, bits: u32) {
        if self.i >= 24 {
            self.i -= bits;
            return;
        }

        self.i -= bits;
        self.prefix >>= bits.saturating_sub(self.i.saturating_sub(24));
        // self.prefix >>= 0.max(bits.min(bits - (self.i - 24)))
    }

    fn advance(&mut self, prefix: u32, bits: u32) {
        if self.i >= 24 {
            self.i += bits;
            return;
        }

        let original_bits = bits;
        let bits = 0.max((24 - self.i).min(bits));

        self.i += original_bits;

        self.prefix <<= bits;
        self.prefix |= prefix >> (original_bits - bits);
    }

    fn can_advance(&mut self, prefix: u32, bits: u32) -> bool {
        let old_i = self.i;
        let old_prefix = self.prefix;

        self.advance(prefix, bits);

        let res = self.check_valid();

        self.i = old_i;
        self.prefix = old_prefix;

        return res;
    }

    fn advance_if_valid(&mut self, prefix: u32, bits: u32) -> bool {
        let old_i = self.i;
        let old_prefix = self.prefix;

        self.advance(prefix, bits);

        let res = self.check_valid();

        if !res {
            self.i = old_i;
            self.prefix = old_prefix;
        }

        return res;
    }

    fn valid_prefixes(&self, bits: u32) -> RangeInclusive<u32> {
        let M = (1 << 24 - self.i - bits) as f64;
        let m = (1 << bits) as f64;

        let L = self.lhs_ignoring_prefix() as f64;
        let R = self.rhs_ignoring_prefix() as f64;

        let lhs = ((L + 1.) / M) - self.prefix as f64 * m - 1.;
        let rhs = (R / M) - self.prefix as f64 * m;

        lhs.ceil() as u32..=rhs.floor() as u32
    }
}

// fn F(b: u8, c: u8) {}

#[derive(Debug)]
struct FreqMap {
    code_to_char: [u8; 256],
    char_to_code: [u8; 256],
}

impl FreqMap {
    fn get_char(&self, code: u8) -> u8 {
        self.code_to_char[code as usize]
    }

    fn get_code(&self, char: u8) -> u8 {
        self.char_to_code[char as usize]
    }

    fn get_code_len(&self, char: u8) -> u8 {
        log2_8[self.get_code(char) as usize] + 1
    }
}

const log2_8: [u8; 256] = [
    0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];

const tab32: [u32; 32] = [
    0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19,
    27, 23, 6, 26, 5, 4, 31,
];

fn u32_size(mut value: u32) -> u32 {
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return tab32[((value.overflowing_mul(0x07C4ACDD).0) as usize) >> 27] + 1;
}

impl FromIterator<(u8, u8)> for FreqMap {
    fn from_iter<T: IntoIterator<Item = (u8, u8)>>(iter: T) -> Self {
        iter.into_iter().fold(
            Self {
                code_to_char: [0; 256],
                char_to_code: [0; 256],
            },
            |mut fm, (code, char)| {
                fm.code_to_char[code as usize] = char;
                fm.char_to_code[char as usize] = code;
                fm
            },
        )
    }
}

fn freq_map(data: &[u8]) -> FreqMap {
    let mut freq_vec = data
        .iter()
        .fold(
            (0..255).map(|x| (x, 0)).collect::<HashMap<_, _>>(),
            |mut x, byte| {
                match x.entry(*byte) {
                    Entry::Occupied(mut o) => *o.get_mut() += 1,
                    Entry::Vacant(v) => *v.insert(0) += 1,
                }

                x
            },
        )
        .into_iter()
        .collect::<Vec<(u8, usize)>>();

    freq_vec.sort_by_key(|(_, ct)| std::cmp::Reverse(*ct));

    freq_vec
        .into_iter()
        .enumerate()
        .map(|(i, (b, _))| (i as u8, b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_encoder() {
        let (table, bytes) = MaxsHuffmanEncoder.pack(b"hello");
    }
}

// chunk dci: 1, bi: 4,  value: 0b 00000001
// chunk dci: 17, bi: 5, value: 0b     0001 0001
// chunk dci: 33, bi: 5, value: 0b           0010 0001 010000011010110100100011
// chunk dci: 50, bi: 4, value: 0b                 001 10010000011010110100100011010
// chunk dci: 32, bi: 4, value: 0b                      00100000110101101001000110100000

// chunk dci: 1, bi: 4, value:  0b0000000100010000 11110100 11110111
// chunk dci: 17, bi: 4, value: 0b    000100010000 11110100 111101110000
// chunk dci: 16, bi: 8, value: 0b        00010000 00001111 0100111101110000
// chunk dci: 0, bi: 4, value:  0b                 00000000 111101001111011100000111

// chunk dci: 1, bi: 4, value:  0b 00000001001001000111010010011001
// chunk dci: 17, bi: 5, value: 0b     00010001001000111010010011001100
// chunk dci: 33, bi: 5, value: 0b          00100001000111010010011001100101
// chunk dci: 32, bi: 5, value: 0b               00100000111010010011001100101000

// finalizing chunk, i: 9, prefix:                          0b1001000
// finalizing chunk, i: 6, prefix:                             0b1000
// finalizing chunk, i: 3, prefix:                                0b0
// finalizing chunk, i: 0, prefix:                                0b0
