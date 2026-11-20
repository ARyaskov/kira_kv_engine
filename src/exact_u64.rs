use crate::{Index, IndexBuilder, IndexError};

/// An MPH and its authoritative u64 keys in slot order. A lookup verifies
/// the entire key, including when the underlying index uses lean mode.
pub struct ExactU64Index {
    index: Index,
    keys_by_slot: Box<[u64]>,
}

impl IndexBuilder {
    /// Build an exact u64 dictionary. The verification table costs eight
    /// bytes per key. Lean mode omits the optional approximate prefilters.
    pub fn build_exact_u64(self, keys: &[u64]) -> Result<ExactU64Index, IndexError> {
        #[cfg(target_endian = "little")]
        let bytes = crate::wire::as_bytes(keys).as_chunks::<8>().0;
        #[cfg(target_endian = "big")]
        let owned: Vec<_> = keys.iter().map(|key| key.to_le_bytes()).collect();
        #[cfg(target_endian = "big")]
        let bytes = owned.as_slice();
        let mut keys_by_slot = vec![0; keys.len()];
        let index = self.build_u64_with_slots(bytes, &mut keys_by_slot)?;
        Ok(ExactU64Index { index, keys_by_slot: keys_by_slot.into_boxed_slice() })
    }
}

impl ExactU64Index {
    /// Return the slot only when its original key equals the query.
    pub fn lookup_u64(&self, key: u64) -> Option<usize> {
        let slot = self.index.lookup_u64(key).ok()?;
        (self.keys_by_slot[slot] == key).then_some(slot)
    }

    /// Check exact membership in input order with bounded stack scratch.
    ///
    /// The prefix for the queried keys is overwritten, and any suffix is preserved.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `keys`.
    pub fn contains_batch_u64_into(&self, keys: &[u64], out: &mut [bool]) {
        const WINDOW: usize = 64;
        assert!(out.len() >= keys.len(), "out slice too small");
        let mut hashes = [0; WINDOW];
        let mut candidates = [None; WINDOW];
        let mut bloom_first = false;
        for (keys, out) in keys.chunks(WINDOW).zip(out[..keys.len()].chunks_mut(WINDOW)) {
            let slots = &mut candidates[..keys.len()];
            self.index.lookup_batch_u64_with_order(keys, &mut hashes, slots, &mut bloom_first);
            for &slot in slots.iter().flatten() {
                crate::prefetch::prefetch_read(&self.keys_by_slot[slot]);
            }
            for ((&key, &slot), result) in keys.iter().zip(slots.iter()).zip(out) {
                *result = slot.is_some_and(|slot| self.keys_by_slot[slot] == key);
            }
        }
    }

    /// Exact membership with no per-query allocation.
    pub fn contains_u64(&self, key: u64) -> bool {
        self.lookup_u64(key).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MphConfig;

    #[test]
    fn exact_slots_match_source_across_parts_and_map_fallback() {
        let keys: Vec<_> =
            (0..40_000u64).map(|key| key.wrapping_mul(0x9e37_79b9_7f4a_7c15)).collect();
        for (input, lambda) in [(&keys[..], 3.0), (&keys[..257], 1_000_000.0)] {
            let config = MphConfig { lambda, seed: 97, max_rehash: 0, ..MphConfig::default() };
            let exact = IndexBuilder::new().with_mph_config(config).build_exact_u64(input).unwrap();
            assert_eq!(exact.index.gpu_export().is_none(), lambda > 3.0);
            let mut seen = vec![false; input.len()];
            for &key in input {
                let slot = exact.lookup_u64(key).unwrap();
                assert!(!std::mem::replace(&mut seen[slot], true));
            }
            assert!(!exact.contains_u64(u64::MAX));
        }
    }

    #[test]
    fn exact_batches_overwrite_only_the_requested_prefix() {
        let queries: Vec<_> = (0..163).chain([u64::MAX, 0]).collect();
        let keys: std::collections::HashSet<_> = queries.iter().step_by(3).copied().collect();
        let input: Vec<_> = keys.iter().copied().collect();
        for lean in [false, true] {
            let exact = IndexBuilder::new().with_lean_mph(lean).build_exact_u64(&input).unwrap();
            let mut result = vec![true; queries.len() + 1];
            exact.contains_batch_u64_into(&queries, &mut result);
            for (key, answer) in queries.iter().zip(&result) {
                assert_eq!(*answer, keys.contains(key));
            }
            assert!(result[queries.len()]);
        }
        let empty = IndexBuilder::new().build_exact_u64(&[]).unwrap();
        let mut result = [true; 2];
        empty.contains_batch_u64_into(&[0], &mut result);
        assert_eq!(result, [false, true]);
    }

    #[test]
    fn exact_membership_rejects_foreign_keys_in_lean_mode() {
        for keys in [&[][..], &[0, 7, u64::MAX][..]] {
            let index = IndexBuilder::new().with_lean_mph(true).build_exact_u64(keys).unwrap();
            for key in [0, 1, 7, 99, u64::MAX] {
                assert_eq!(index.contains_u64(key), keys.contains(&key));
            }
        }
    }
}
