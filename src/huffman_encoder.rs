use crate::huffman::HuffmanTable;

use bitvec::prelude::*;
use huffman_coding::HuffmanTree;
use std::collections::HashMap;

struct HuffmanEncoder {
    table: HuffmanTable,
    compressed: Vec<u8>,
}

impl HuffmanEncoder {
    pub fn pack(&mut self, data: &[u8]) {
        let mut weights = HashMap::new();
        for byte in data {
            *weights.entry(*byte).or_insert(0) += 1;
        }

        let tree = huffman_coding::HuffmanTree::from_data(data);
        self.generate_codes(tree, BitVec::new());

        for byte in data {
            let (item, _, len) = self.table.code_dict[*byte as usize];
            self.compressed
                .extend_from_slice(&item.to_be_bytes()[..len as usize]);
        }
    }

    fn generate_codes(&mut self, node: HuffmanTree, code: BitVec<u8>) {
        match node {
            HuffmanTree::Leaf(item, prob) => {
                self.table.code_dict[code.load::<u8>() as usize] = (item, true, code.len() as u32);
            }
            HuffmanTree::Node(left, right) => {
                let mut left_code = code.clone();
                left_code.push(false);
                self.generate_codes(*left, left_code);
                let mut right_code = code.clone();
                right_code.push(true);
                self.generate_codes(*right, right_code);
            }
        }
    }

    pub fn finish(self) -> (HuffmanTable, Vec<u8>) {
        (self.table, self.compressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_encoder() {
        let mut encoder = HuffmanEncoder {
            table: HuffmanTable::default(),
            compressed: vec![],
        };
        let data = b"hello world";
        encoder.pack(data);
        let (table, packed) = encoder.finish();
        assert_eq!(packed, vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]);
        assert_eq!(table.dictionary.len(), 5);
        assert_eq!(table.code_dict.len(), 256);
        assert_eq!(table.min_codes.len(), 33);
        assert_eq!(table.max_codes.len(), 33);
    }
}
