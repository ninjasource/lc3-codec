#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bandwidth {
    /// 0 - Narrow Band (4khz audio bandwidth)
    NarrowBand = 0,

    /// 1 - Wide Band (8khz audio bandwidth)
    WideBand = 1,

    /// 2 - Semi Super Wide Band (12khz audio bandwidth)
    SemiSuperWideBand = 2,

    /// 3 - Super Wide Band (16khz audio bandwidth)
    SuperWideBand = 3,

    /// 4 - Full Band (20khz audio bandwidth)
    FullBand = 4,
}

#[derive(Debug)]
pub struct SideInfo {
    // in the order in which they are decoded
    pub bandwidth: Bandwidth,                     // bandwidth cutoff index (P_BW)
    pub lastnz: usize,                            // last non-zero tupple
    pub lsb_mode: bool,                           // lsb mode bit (least significant bit)
    pub global_gain_index: usize,                 // global gain index (gg_ind or gg_idx)
    pub num_tns_filters: usize,                   // number of temporal noise shaping (tns) filters
    pub reflect_coef_order_ari_input: [usize; 2], // Intermediate order of quantized reflection coefficients used as input to the arithmetic decoder (further information is extracted there)
    pub sns_vq: SnsVq,
    pub long_term_post_filter_info: LongTermPostFilterInfo,
    pub noise_factor: usize, // noise level - noise factor (f_nf)
}

#[derive(Debug, Clone, Copy)]
pub struct LongTermPostFilterInfo {
    pub pitch_present: bool, // pitch present flag
    // only set if pitch_present = true
    pub is_active: bool,    // is long-term post filter active
    pub pitch_index: usize, // long-term post filter pitch index (lookup for the pitch lag)
}

impl LongTermPostFilterInfo {
    pub fn new(is_active: bool, pitch_present: bool, pitch_index: usize) -> Self {
        Self {
            is_active,
            pitch_present,
            pitch_index,
        }
    }
}

#[derive(Debug)]
pub struct SnsVq {
    pub ind_lf: usize,
    pub ind_hf: usize,
    pub ls_inda: usize,
    pub ls_indb: usize,
    pub idx_a: usize,
    pub idx_b: usize,
    pub submode_lsb: u8,
    pub submode_msb: u8,
    pub g_ind: usize,
}
