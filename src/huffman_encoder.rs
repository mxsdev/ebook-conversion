use crate::huffman::HuffmanTable;

use bitvec::prelude::*;
use huffman_coding::HuffmanTree;
use std::{collections::HashMap, io::Write};

struct HuffmanEncoder {
    table: HuffmanTable,
    byte_to_code: HashMap<u8, BitVec<u8>>,
    compressed: BitVec<u8, Msb0>,
}

impl HuffmanEncoder {
    pub fn pack(&mut self, data: &[u8]) {
        let mut weights = HashMap::new();
        for byte in data {
            *weights.entry(*byte).or_insert(0) += 1;
        }

        let tree = huffman_coding::HuffmanTree::from_data(data);
        // self.generate_codes(&tree, BitVec::new());
        // self.generate_min_max_depths();
        // self.build_dictionary(&tree, BitVec::new());
        self.generate_byte_to_code_mapping(&tree, BitVec::new());

        // Temporary hack to make all codes uniform length and unique
        let mut i: u8 = 1;
        for (_, code) in self.byte_to_code.iter_mut() {
            *code = (BitVec::from_element(i)[0..4]).to_bitvec();
            i += 1;
        }

        println!("{:?}", self.byte_to_code);

        for i in 0..256 {
            self.table.code_dict[i] = (4, true, u32::MAX);
        }

        for byte in data {
            let code = self.byte_to_code.get(byte).unwrap();

            let dictionary_index = code.load::<u8>(); // << (8 - code.len());

            println!("dictionary index: {}", dictionary_index);
            self.table.dictionary[dictionary_index as usize] = Some((vec![*byte], true));

            let shifted_code = (dictionary_index as u32) << (32 - code.len());
            let mut partial_code: u32 = u32::MAX - shifted_code;

            // if self.compressed.len() % 8 != 0 {
            //     let previous_unused_code = self
            //         .compressed
            //         .iter()
            //         .rev()
            //         .take((8 - (self.compressed.len() % 8)))
            //         .collect::<BitVec>()
            //         .load::<u8>();

            //     // partial_code -= previous_unused_code;
            // }

            println!("partial code: {:b}", partial_code);
            let partial_code: BitVec<_, Msb0> = BitVec::from_element(partial_code);
            println!("sliced: {:?}", partial_code[0..code.len()].to_bitvec());

            self.compressed
                .append(&mut partial_code[0..code.len()].to_bitvec());
            println!("compressed: {:b}", self.compressed.load::<u64>());
        }

        // Padding
        self.table.code_dict[0] = (8, true, 0);
        // self.table.dictionary[0] = Some((vec![], true));
    }

    fn generate_byte_to_code_mapping(&mut self, node: &HuffmanTree, current_code: BitVec<u8>) {
        match node {
            HuffmanTree::Leaf(item, prob) => {
                self.byte_to_code.insert(*item, current_code);
            }
            HuffmanTree::Node(left, right) => {
                let mut left_code = current_code.clone();
                left_code.push(false);
                self.generate_byte_to_code_mapping(left, left_code);
                let mut right_code = current_code.clone();
                right_code.push(true);
                self.generate_byte_to_code_mapping(right, right_code);
            }
        }
    }

    fn generate_codes(&mut self, node: &HuffmanTree, code: BitVec<u8>) {
        match node {
            HuffmanTree::Leaf(item, prob) => {
                self.table.code_dict[*item as usize] =
                    (code.len() as u8, true, code.load::<u32>() + *item as u32);
            }
            HuffmanTree::Node(left, right) => {
                let mut left_code = code.clone();
                left_code.push(false);
                self.generate_codes(left, left_code);
                let mut right_code = code.clone();
                right_code.push(true);
                self.generate_codes(right, right_code);
            }
        }
    }

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

    fn build_dictionary(&mut self, node: &HuffmanTree, prefix: BitVec<u8>) {
        match node {
            HuffmanTree::Leaf(item, _) => {
                let flag = prefix.is_empty();
                // todo: how to remove + 1 / padding?
                self.table.dictionary[prefix.load::<u8>() as usize + 1] = Some((vec![*item], true));
            }
            HuffmanTree::Node(left, right) => {
                let mut left_prefix = prefix.clone();
                left_prefix.push(false);
                self.build_dictionary(left, left_prefix);

                let mut right_prefix = prefix.clone();
                right_prefix.push(true);
                self.build_dictionary(right, right_prefix);
            }
        }
    }

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
        let data = b"heyyasdfasdofi";
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
