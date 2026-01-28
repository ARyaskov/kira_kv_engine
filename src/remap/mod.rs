use crate::pgm::PgmIndex;

pub fn remap_ids_for_pgm(pgm: &PgmIndex) -> Vec<u64> {
    let mut out = Vec::with_capacity(pgm.keys_len());
    let mut seg_id = 0u64;
    while let Some((start, end)) = pgm.segment_bounds(seg_id as usize) {
        let mut local = 0u64;
        let end_u = end as usize;
        let mut i = start as usize;
        while i < end_u {
            let remap = (seg_id << 32) | local;
            out.push(remap);
            local += 1;
            i += 1;
        }
        seg_id += 1;
    }
    out
}

#[inline]
pub fn remap_id_from_index(pgm: &PgmIndex, seg_id: usize, global_index: usize) -> u64 {
    if let Some((start, _end)) = pgm.segment_bounds(seg_id) {
        let local = global_index.saturating_sub(start as usize) as u64;
        ((seg_id as u64) << 32) | local
    } else {
        (seg_id as u64) << 32
    }
}
