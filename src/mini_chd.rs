//! Mini-CHD — minimal perfect hash for **small** key sets (≤ 4096 keys).
//!
//! Designed for use inside `HybridIndex` per-segment storage where we have
//! thousands of independent tiny MPHs to build. PtrHash25's machinery
//! (prerotation learning, 2-level bucketing, compressed pilots, build arena)
//! adds 5–10 ms of fixed overhead per instance — too much when summed over
//! 50K segments. MiniChd strips all that to the bare CHD essentials:
//!
//! - **One hash function** (splitmix64) with a per-instance salt.
//! - **Single-level bucketing**: bucket = hash(key, salt) % num_buckets.
//! - **u8 pilots**: per-bucket displacement, tried 0..255 greedily.
//! - **Near-minimal**: slot space = `ceil(1.10 · N)` so the last-bucket
//!   pilot search converges in 1–2 attempts.
//!
//! Memory: ~1.1 bytes/key (one u8 per bucket, gamma = ~0.5).
//! Build:  ~50–500 µs for 100–4000 keys (vs 2–10 ms for PtrHash25).
//! Lookup: ~6 ns (one splitmix + one mod + one u8 read).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MiniChdError {
    #[error("unbuildable: a bucket exceeded pilot space; rebuild with a different seed")]
    Unresolvable,
    #[error("empty key set")]
    Empty,
}

/// Compact MPH for a small u64 key set.
#[derive(Debug, Clone)]
pub struct MiniChd {
    /// Slot space (= ceil(1.10 · N) by default).
    pub n: u32,
    /// Per-instance hash salt.
    pub salt: u64,
    /// Pilot per bucket. `pilots.len()` == `num_buckets`.
    pub pilots: Box<[u8]>,
    /// Number of buckets (target ~ 2× keys for low collision rate per bucket).
    pub num_buckets: u32,
}

impl MiniChd {
    /// Build a MiniChd over the given u64 keys. Returns `None` if all 16 salt
    /// retries fail (extremely rare for clean inputs).
    pub fn build(keys: &[u64], base_seed: u64) -> Result<Self, MiniChdError> {
        if keys.is_empty() {
            return Err(MiniChdError::Empty);
        }
        let n = ((keys.len() as f64) * 1.10).ceil() as usize;
        // gamma = 0.5: 2 slots per bucket on average. Tighter packing → harder
        // pilot search; this gives 1-2 attempts per bucket typically.
        let num_buckets = ((n as f64) * 0.5).ceil().max(1.0) as usize;

        for retry in 0..16u64 {
            let salt = splitmix64(base_seed ^ retry.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            if let Some(mphf) = Self::try_build_with_salt(keys, salt, n, num_buckets) {
                return Ok(mphf);
            }
        }
        Err(MiniChdError::Unresolvable)
    }

    fn try_build_with_salt(
        keys: &[u64],
        salt: u64,
        n: usize,
        num_buckets: usize,
    ) -> Option<Self> {
        // Group keys by bucket.
        let mut buckets: Vec<Vec<u64>> = vec![Vec::new(); num_buckets];
        for &k in keys {
            let b = (splitmix64(k ^ salt) % num_buckets as u64) as usize;
            buckets[b].push(k);
        }

        // Process largest buckets first — they're the hardest to place.
        let mut order: Vec<usize> = (0..num_buckets).collect();
        order.sort_unstable_by_key(|&b| std::cmp::Reverse(buckets[b].len()));

        let mut pilots = vec![0u8; num_buckets];
        let mut used = vec![false; n];

        for b in order {
            let bk = &buckets[b];
            if bk.is_empty() {
                continue;
            }
            let mut found = false;
            for p in 0..=255u8 {
                // Try this pilot — does every key in the bucket land on a unique
                // unused slot?
                let mut placement = Vec::with_capacity(bk.len());
                let mut ok = true;
                for &k in bk {
                    let slot = slot_for(k, salt, p, n);
                    if used[slot] || placement.contains(&slot) {
                        ok = false;
                        break;
                    }
                    placement.push(slot);
                }
                if ok {
                    pilots[b] = p;
                    for s in placement {
                        used[s] = true;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }

        Some(Self {
            n: n as u32,
            salt,
            pilots: pilots.into_boxed_slice(),
            num_buckets: num_buckets as u32,
        })
    }

    /// O(1) lookup. Caller verifies via fingerprint if needed.
    #[inline]
    pub fn index(&self, key: u64) -> u32 {
        let bucket = (splitmix64(key ^ self.salt) % self.num_buckets as u64) as usize;
        let pilot = unsafe { *self.pilots.get_unchecked(bucket) };
        slot_for(key, self.salt, pilot, self.n as usize) as u32
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.pilots.len()
    }
}

#[inline]
fn slot_for(key: u64, salt: u64, pilot: u8, n: usize) -> usize {
    let mixed = splitmix64(key ^ salt ^ ((pilot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    (mixed % n as u64) as usize
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

