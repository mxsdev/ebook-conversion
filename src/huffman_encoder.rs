use crate::huffman::HuffmanTable;

use bitvec::prelude::*;
use huffman_coding::HuffmanTree;
use std::{collections::HashMap, io::Write};

struct HuffmanEncoder {
    table: HuffmanTable,
    compressed: Vec<u8>,
}

impl HuffmanEncoder {
    pub fn pack(&mut self, data: &[u8]) {
        // let mut weights = HashMap::new();
        // for byte in data {
        //     *weights.entry(*byte).or_insert(0) += 1;
        // }

        // let tree = huffman_coding::HuffmanTree::from_data(data);
        // self.generate_codes(&tree, BitVec::new());
        // self.generate_min_max_depths();
        // self.build_dictionary(&tree, Vec::new());

        // println!("{:?}", self.table.max_codes);

        // let mut writer = huffman_coding::HuffmanWriter::new(&mut self.compressed, &tree);
        // writer.write_all(data).unwrap();


        let mut n = 32i8;
        for byte in data {
            let code = ((*byte as u32) << 24) << (n as u32);
            self.table.code_dict[*byte as usize] = (8 as u8, true, code);
            self.compressed.push(*byte);
            self.table.dictionary[0] = Some((vec![*byte], true));

            self.table.min_codes[8] = self.table.min_codes[8].min(code);
            self.table.max_codes[8] = self.table.max_codes[8].max(code);
        }

        // for byte in data {
        //     let (item, _, len) = self.table.code_dict[*byte as usize];
        //     println!("{:?}", &item.to_be_bytes()[..len as usize]);
        //     self.compressed
        //         .extend_from_slice(&item.to_be_bytes()[..len as usize]);
        // }

        // println!("{:?}", self.compressed);
    }

    fn generate_codes(&mut self, node: &HuffmanTree, code: BitVec<u8>) {
        match node {
            HuffmanTree::Leaf(item, prob) => {
                self.table.code_dict[*item as usize] =
                    (code.len() as u8, true, code.load::<u32>());
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

    fn build_dictionary(&mut self, node: &HuffmanTree, prefix: Vec<u8>) {
        match node {
            HuffmanTree::Leaf(item, _) => {
                let flag = prefix.is_empty();
                self.table.dictionary[*item as usize] = Some((prefix.clone(), flag));
            }
            HuffmanTree::Node(left, right) => {
                let mut left_prefix = prefix.clone();
                left_prefix.push(0);
                self.build_dictionary(left, left_prefix);

                let mut right_prefix = prefix.clone();
                right_prefix.push(1);
                self.build_dictionary(right, right_prefix);
            }
        }
    }

    pub fn finish(self) -> (HuffmanTable, Vec<u8>) {
        (self.table, self.compressed)
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
            compressed: vec![],
        };
        let data = b"h";
        encoder.pack(data);
        let (table, packed) = encoder.finish();
        // assert_eq!(packed, vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]);
        // assert_eq!(table.dictionary.len(), 5);
        // assert_eq!(table.code_dict.len(), 256);
        // assert_eq!(table.min_codes.len(), 33);
        // assert_eq!(table.max_codes.len(), 33);

        let mut decoder = HuffmanDecoder { table };

        let unpacked = decoder.unpack(&packed);
        assert_eq!(unpacked, data);
    }
}
