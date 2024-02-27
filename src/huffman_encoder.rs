use crate::huffman::HuffmanTable;

use bitvec::prelude::*;
// use huffman_coding::HuffmanTree;
use huffman_compress::{Book, CodeBuilder, Tree};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HuffmanEncodingError {
    // todo: shouldn't need this error
    #[error("Not enough unique data")]
    NotEnoughUniqueData,
}

pub struct HuffmanEncoder {
    table: HuffmanTable,
    code_length_to_max_code: HashMap<usize, u32>,
    byte_to_code: HashMap<u16, BitVec<u8>>,
    compressed: BitVec<u8, Msb0>,
}

impl Default for HuffmanEncoder {
    fn default() -> Self {
        Self {
            table: HuffmanTable::default(),
            code_length_to_max_code: HashMap::new(),
            byte_to_code: HashMap::new(),
            compressed: BitVec::new(),
        }
    }
}

impl HuffmanEncoder {
    pub fn pack(&mut self, data: &[u8]) -> Result<(), HuffmanEncodingError> {
        println!("data: {:?}", data);
        let mut weights: HashMap<u16, i32> = HashMap::new();
        for byte in data {
            *weights.entry(*byte as u16).or_insert(0) += 1;
        }

        // the actual value doesn't matter here, it won't collide with anything because it's u16 vs u8
        let null_padding_symbol: u16 = 0xffff;
        weights.insert(null_padding_symbol, 1);

        // todo: remove
        if weights.len() < 2 {
            return Err(HuffmanEncodingError::NotEnoughUniqueData);
        }

        let (book, tree) = CodeBuilder::from_iter(weights).finish();

        for symbol in book.symbols() {
            let code = book.get(symbol).unwrap();
            let mut parsed_code: BitVec<u8> = BitVec::new();
            for bit in code {
                parsed_code.push(bit);
            }

            self.byte_to_code.insert(*symbol, parsed_code);
        }

        // let mut sorted_byte_to_code: Vec<_> = self.byte_to_code.iter().collect();
        // sorted_byte_to_code.sort_by(|a, b| a.1.len().cmp(&b.1.len()));

        // for (i, (_, code)) in sorted_byte_to_code.iter().enumerate() {
        //     self.table.code_dict[self.table.code_dict.len() - 1 - i] =
        //         (code.len() as u8, true, u32::MAX);
        // }

        // let codes_to_write = data
        //     .iter()
        //     .map(|byte| (*self.byte_to_code.get(&(*byte as u16)).unwrap()).clone())
        //     .collect::<Vec<BitVec<u8>>>();

        // for code in codes_to_write {
        //     let last_dangling_bits =
        //         &self.compressed[self.compressed.len() - (self.compressed.len() % 8)..];

        // }

        // loop {
        //     let mut pending_codes_to_write: Vec<BitVec<u8>> = Vec::new();

        //     while pending_codes_to_write
        //         .iter()
        //         .map(|code| code.len())
        //         .sum::<usize>()
        //         < 8
        //     {
        //         pending_codes_to_write.push(codes_to_write_reverse.pop().unwrap());
        //     }

        //     println!("pending: {:?}", pending_codes_to_write);
        //     break;
        // }

        // todo!();

        let unique_code_lengths = self
            .byte_to_code
            .values()
            .map(|code| code.len())
            .collect::<HashSet<_>>();

        for unique_code_length in unique_code_lengths {
            self.compute_max_code_for_length(unique_code_length);
        }

        println!("max code: {:?}", self.code_length_to_max_code);

        for (byte, code) in self.byte_to_code.clone() {
            println!("byte: {}, code: {:?}", byte, code);
        }

        let mut pending_code_dict_entries: Vec<u8> = vec![];

        for (i, byte) in data.iter().enumerate() {
            let code = self.byte_to_code.get(&(*byte as u16)).unwrap();

            let dictionary_index = code.load::<u8>(); // << (8 - code.len());

            self.table.dictionary[dictionary_index as usize] = Some((vec![*byte], true));

            let shifted_code = (dictionary_index as u32) << (32 - code.len());
            let partial_code: u32 =
                self.code_length_to_max_code.get(&code.len()).unwrap() - shifted_code;
            let partial_code: BitVec<_, Msb0> = BitVec::from_element(partial_code);

            self.compressed
                .append(&mut partial_code[0..code.len()].to_bitvec());

            pending_code_dict_entries.push(code.len() as u8);
        }

        // Padding
        // todo: combine with above
        let code = self.byte_to_code.get(&null_padding_symbol).unwrap().clone();

        let dictionary_index = code.load::<u8>(); // << (8 - code.len());

        println!("dictionary index: {}", dictionary_index);

        let len = 64 - self.compressed.len() as u8;
        self.compute_max_code_for_length(len as usize);

        let shifted_code = (dictionary_index as u32) << (32 - code.len());
        let partial_code: u32 =
            self.code_length_to_max_code.get(&(len as usize)).unwrap() - shifted_code;
        let partial_code: BitVec<_, Msb0> = BitVec::from_element(partial_code);

        self.compressed
            .append(&mut partial_code[0..code.len()].to_bitvec());

        pending_code_dict_entries.push(len);
        /* end of padding */

        for (len, max_code) in self.code_length_to_max_code.iter() {
            println!("max code for length {:02}: {:032b}", len, max_code);
        }

        let mut pos = 0;
        for len in pending_code_dict_entries {
            let last_8_bits = self.compressed[pos..self.compressed.len().min(pos + 8)].to_bitvec();
            println!("last 8 bits before shift: {:?}", last_8_bits);

            let num_of_bits = last_8_bits.len();
            let mut last_8_bits = last_8_bits.load_be::<u8>();

            // todo: necessary?
            if num_of_bits < 8 {
                last_8_bits = last_8_bits << (8 - num_of_bits);
            }

            println!(
                "last 8 bits after shif: {:08b}, length: {}",
                last_8_bits, len
            );

            if self.table.code_dict[last_8_bits as usize].0 != 0
                && self.table.code_dict[last_8_bits as usize].0 != len
            {
                panic!("collision");
            }

            self.table.code_dict[last_8_bits as usize] = (
                len,
                true,
                *self.code_length_to_max_code.get(&(len as usize)).unwrap(),
            );
            pos += len as usize;
        }

        for pair in self.table.code_dict.iter().enumerate() {
            if (*pair.1).0 != 0 {
                println!("code dict {:?}, bit index: {:08b}", pair, pair.0);
            }
        }

        Ok(())
    }

    fn compute_max_code_for_length(&mut self, len: usize) {
        let mut max_code = u32::MAX;

        let mut codes_to_check = self
            .byte_to_code
            .iter()
            // .filter(|(_, code)| code.len() == len)
            .collect::<Vec<_>>();

        if codes_to_check.is_empty() {
            println!("adding code for null padidng symbol {}", len);
            let null_padding_symbol: u16 = 0xffff;
            // codes_to_check = vec![(&0, self.byte_to_code.get(&null_padding_symbol).unwrap())];
            codes_to_check.push((&0, self.byte_to_code.get(&null_padding_symbol).unwrap()));
        }

        loop {
            let unique_results = codes_to_check
                .iter()
                .map(|(_, code)| max_code - (code.load::<u8>() as u32))
                .collect::<HashSet<u32>>();

            if unique_results.len() != codes_to_check.len()
                || self
                    .code_length_to_max_code
                    .values()
                    .any(|&v| v == max_code)
            {
                // Subtract 1 from the top 8 bits
                let top_8_bits_modified = ((max_code & 0xFF000000) >> 24) - 1;
                let top_8_bits_modified = (top_8_bits_modified << 24);
                let max_code_cleared_top_8_bits = max_code & 0x00ffffff;
                max_code = top_8_bits_modified | max_code_cleared_top_8_bits;
            } else {
                break;
            }
        }

        self.code_length_to_max_code.insert(len, max_code);
    }

    // fn generate_byte_to_code_mapping(&mut self, node: &HuffmanTree, current_code: BitVec<u8>) {
    //     match node {
    //         HuffmanTree::Leaf(item, prob) => {
    //             // todo: avoid +1
    //             println!("code: {:?} {:?}", current_code, node);
    //             let current_code = BitVec::from_element(current_code.load::<u8>() + 1);
    //             self.byte_to_code.insert(*item, current_code);
    //         }
    //         HuffmanTree::Node(left, right) => {
    //             let mut left_code = current_code.clone();
    //             left_code.push(false);
    //             self.generate_byte_to_code_mapping(left, left_code);
    //             let mut right_code = current_code.clone();
    //             right_code.push(true);
    //             self.generate_byte_to_code_mapping(right, right_code);
    //         }
    //     }
    // }

    // fn generate_codes(&mut self, node: &HuffmanTree, code: BitVec<u8>) {
    //     match node {
    //         HuffmanTree::Leaf(item, prob) => {
    //             self.table.code_dict[*item as usize] =
    //                 (code.len() as u8, true, code.load::<u32>() + *item as u32);
    //         }
    //         HuffmanTree::Node(left, right) => {
    //             let mut left_code = code.clone();
    //             left_code.push(false);
    //             self.generate_codes(left, left_code);
    //             let mut right_code = code.clone();
    //             right_code.push(true);
    //             self.generate_codes(right, right_code);
    //         }
    //     }
    // }

    fn generate_min_max_depths(&mut self) {
        for (code_len, term, code) in self.table.code_dict {
            // Skip terminal codes
            // if (term) {
            //     continue;
            // }

            self.table.min_codes[code_len as usize] =
                self.table.min_codes[code_len as usize].min(code);
            self.table.max_codes[code_len as usize] =
                self.table.max_codes[code_len as usize].max(code);
        }

        // Adjust max codes for non-terminal codes
        // for i in 0..self.table.max_codes.len() {
        //     if (self.table.max_codes[i] != u32::MAX) {
        //         self.table.max_codes[i] =
        //             (self.table.max_codes[i] << (32 - i)) | ((1 << (32 - i)) - 1)
        //     }
        // }
    }

    // fn build_dictionary(&mut self, node: &HuffmanTree, prefix: BitVec<u8>) {
    //     match node {
    //         HuffmanTree::Leaf(item, _) => {
    //             let flag = prefix.is_empty();
    //             // todo: how to remove + 1 / padding?
    //             self.table.dictionary[prefix.load::<u8>() as usize + 1] = Some((vec![*item], true));
    //         }
    //         HuffmanTree::Node(left, right) => {
    //             let mut left_prefix = prefix.clone();
    //             left_prefix.push(false);
    //             self.build_dictionary(left, left_prefix);

    //             let mut right_prefix = prefix.clone();
    //             right_prefix.push(true);
    //             self.build_dictionary(right, right_prefix);
    //         }
    //     }
    // }

    pub fn finish(self) -> (HuffmanTable, Vec<u8>) {
        (self.table, self.compressed.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use crate::huffman_decoder::HuffmanDecoder;

    use super::*;

    #[test]
    fn test_huffman_encoder() {
        let mut encoder = HuffmanEncoder {
            table: HuffmanTable::default(),
            compressed: BitVec::new(),
            byte_to_code: HashMap::new(),
            ..Default::default()
        };
        let data = b"hey";
        encoder.pack(data);
        let (table, packed) = encoder.finish();
        // assert_eq!(packed, vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]);
        // assert_eq!(table.dictionary.len(), 5);
        // assert_eq!(table.code_dict.len(), 256);
        // assert_eq!(table.min_codes.len(), 33);
        // assert_eq!(table.max_codes.len(), 33);

        let mut decoder = HuffmanDecoder { table };

        let unpacked = decoder.unpack(&packed);
        println!("unpacked: {}", String::from_utf8_lossy(&unpacked));
        assert_eq!(unpacked, data);
    }
}
