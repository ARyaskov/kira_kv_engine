//! Independent correctness checks shared by benchmark examples and their tests.
use kira_kv_engine::Index;

pub struct SlotOracle {
    source_by_slot: Vec<u32>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct QueryStats {
    pub queries: usize,
    pub known_hits: usize,
    pub foreign_queries: usize,
    pub foreign_candidates: usize,
}

/// Build an authoritative slot-to-full-key binding. Arbitrary MPH permutations
/// are valid, but duplicate/out-of-range slots and missing known keys are not.
pub fn slot_oracle<K: AsRef<[u8]>>(
    keys: &[K],
    slots: usize,
    mut lookup: impl FnMut(&[u8]) -> Option<usize>,
) -> Result<SlotOracle, String> {
    if keys.len() > u32::MAX as usize || slots < keys.len() {
        return Err("invalid oracle slot capacity".into());
    }
    let mut source_by_slot = vec![u32::MAX; slots];
    for (source, key) in keys.iter().enumerate() {
        let slot = lookup(key.as_ref()).ok_or("known key was rejected")?;
        let binding = source_by_slot.get_mut(slot).ok_or("lookup returned an invalid slot")?;
        if *binding != u32::MAX {
            return Err("two distinct input keys share an MPH slot".into());
        }
        *binding = source as u32;
    }
    Ok(SlotOracle { source_by_slot })
}

pub fn verify_index<K: AsRef<[u8]>>(index: &Index, keys: &[K]) -> Result<SlotOracle, String> {
    if index.len() != keys.len() {
        return Err("index key count differs from the input".into());
    }
    slot_oracle(keys, index.slot_capacity(), |key| index.lookup(key).ok())
}

/// Check actual batch results against the independently bound original bytes.
/// Foreign candidates are measured, not treated as exact-membership failures.
pub fn verify_results<'a, K: AsRef<[u8]>>(
    oracle: &SlotOracle,
    keys: &[K],
    queries: impl IntoIterator<Item = (&'a [u8], bool)>,
    results: &[Option<usize>],
) -> Result<QueryStats, String> {
    let mut stats = QueryStats::default();
    let mut queries = queries.into_iter();
    for result in results {
        let (key, known) = queries.next().ok_or("batch has too many results")?;
        stats.queries += 1;
        if known {
            stats.known_hits += 1;
        } else {
            stats.foreign_queries += 1;
        }
        match *result {
            Some(slot) => {
                let source =
                    *oracle.source_by_slot.get(slot).ok_or("batch returned an invalid slot")?;
                if known {
                    let original = keys.get(source as usize).ok_or("batch hit an unbound slot")?;
                    if original.as_ref() != key {
                        return Err("batch returned the slot of another known key".into());
                    }
                } else {
                    stats.foreign_candidates += 1;
                }
            }
            None if known => return Err("batch rejected a known key".into()),
            None => {}
        }
    }
    if queries.next().is_some() {
        return Err("batch has too few results".into());
    }
    Ok(stats)
}
