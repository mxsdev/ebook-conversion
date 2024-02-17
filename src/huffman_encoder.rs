use crate::huffman::HuffmanTable;

use bitvec::prelude::*;
// use huffman_coding::HuffmanTree;
use huffman_compress::{Book, CodeBuilder, Tree};
use std::{collections::HashMap, io::Write};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HuffmanEncodingError {
    // todo: shouldn't need this error
    #[error("Not enough unique data")]
    NotEnoughUniqueData,
}

pub struct HuffmanEncoder {
    table: HuffmanTable,
    byte_to_code: HashMap<u16, BitVec<u8>>,
    compressed: BitVec<u8, Msb0>,
}

impl Default for HuffmanEncoder {
    fn default() -> Self {
        Self {
            table: HuffmanTable::default(),
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

        // the actual value doesn't matter here, it won't collide with anything
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

        // println!(
        //     "{:?}",
        //     book.symbols()
        //         .map(|sym| (sym, book.get(sym).unwrap().len()))
        //         .collect::<Vec<_>>()
        // );

        // for i in 0..256 {
        //     self.table.code_dict[i] = (4, true, u32::MAX);
        // }

        let mut pending_code_dict_entries: Vec<u8> = vec![];

        for (i, byte) in data.iter().enumerate() {
            let code = self.byte_to_code.get(&(*byte as u16)).unwrap();

            let dictionary_index = code.load::<u8>(); // << (8 - code.len());

            println!("dictionary index: {}", dictionary_index);
            self.table.dictionary[dictionary_index as usize] = Some((vec![*byte], true));

            let shifted_code = (dictionary_index as u32) << (32 - code.len());
            let partial_code: u32 = u32::MAX - shifted_code;
            let partial_code: BitVec<_, Msb0> = BitVec::from_element(partial_code);

            self.compressed
                .append(&mut partial_code[0..code.len()].to_bitvec());

            pending_code_dict_entries.push(code.len() as u8);
        }

        // Padding
        // todo: combine with above
        let code = self.byte_to_code.get(&null_padding_symbol).unwrap();

        let dictionary_index = code.load::<u8>(); // << (8 - code.len());

        println!("dictionary index: {}", dictionary_index);

        let shifted_code = (dictionary_index as u32) << (32 - code.len());
        let partial_code: u32 = u32::MAX - shifted_code;
        let partial_code: BitVec<_, Msb0> = BitVec::from_element(partial_code);

        self.compressed
            .append(&mut partial_code[0..code.len()].to_bitvec());

        pending_code_dict_entries.push(64 - self.compressed.len() as u8);

        let mut pos = 0;
        for len in pending_code_dict_entries {
            let last_8_bits = self.compressed[pos..self.compressed.len().min(pos + 8)].to_bitvec();
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

            self.table.code_dict[last_8_bits as usize] = (len, true, u32::MAX);
            pos += len as usize;
        }

        for pair in self.table.code_dict.iter().enumerate() {
            if (*pair.1).0 != 0 {
                println!("code dict {:?}, bit index: {:08b}", pair, pair.0);
            }
        }

        Ok(())
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
