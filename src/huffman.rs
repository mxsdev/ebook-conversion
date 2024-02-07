type HuffmanDictionary = Vec<Option<(Vec<u8>, bool)>>;
type CodeDictionary = [(u8, bool, u32); 256];
type MinCodesMapping = [u32; 33];
type MaxCodesMapping = [u32; 33];

pub struct HuffmanTable {
    pub dictionary: HuffmanDictionary,
    pub code_dict: CodeDictionary,
    pub min_codes: MinCodesMapping,
    pub max_codes: MaxCodesMapping,
}

impl Default for HuffmanTable {
    fn default() -> Self {
        Self {
            dictionary: vec![None; 256],
            code_dict: [(0, false, 0); 256],
            min_codes: [u32::MAX; 33],
            max_codes: [0; 33],
        }
    }
}
