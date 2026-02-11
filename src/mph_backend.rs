use hashbrown::{HashMap, HashSet};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    PTHash,
    PtrHash2025,
    CHD,
    RecSplit,
    #[cfg(feature = "bbhash")]
    BBHash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildProfile {
    Balanced,
    Fast,
}

#[derive(Debug, Clone)]
pub struct BuildConfig {
    pub backend: BackendKind,
    pub hot_fraction: f32,
    pub hot_backend: BackendKind,
    pub cold_backend: BackendKind,
    pub enable_parallel_build: bool,
    pub seed: u64,
    pub gamma: f64,
    pub rehash_limit: u32,
    pub max_pilot_attempts: u32,
    pub build_profile: BuildProfile,
    pub fast_fail_rounds: u32,
    pub frequencies: Option<Vec<(u64, u32)>>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::PtrHash2025,
            hot_fraction: 0.15,
            hot_backend: BackendKind::CHD,
            cold_backend: BackendKind::RecSplit,
            enable_parallel_build: true,
            seed: 0xC0FF_EE00_D15E_A5E,
            gamma: 1.2,
            rehash_limit: 10,
            max_pilot_attempts: 8_192,
            build_profile: BuildProfile::Balanced,
            fast_fail_rounds: 2,
            frequencies: None,
        }
    }
}

pub trait MphBackend {
    fn build(keys: &[u64], config: &BuildConfig) -> Self
    where
        Self: Sized;

    fn lookup(&self, key: u64) -> Option<u32>;

    fn memory_usage_bytes(&self) -> usize;
}

#[derive(Debug, Clone)]
struct CoreMph {
    n: u32,
    m: u32,
    seed: u64,
    hash_s1: u64,
    hash_s2: u64,
    hash_s3: u64,
    pilots: Vec<u32>,
}

impl CoreMph {
    fn build(keys: &[u64], cfg: &BuildConfig, gamma: f64, pilot_budget: u32) -> Option<Self> {
        if keys.is_empty() {
            return None;
        }

        let n = keys.len();
        let gamma_schedule = build_gamma_schedule(cfg.build_profile, gamma);
        let max_rounds = cfg.rehash_limit.max(1);

        for (stage, stage_gamma) in gamma_schedule.into_iter().enumerate() {
            let buckets = ((n as f64 / stage_gamma.max(0.5)).ceil() as usize).max(1);
            let stage_rounds = if matches!(cfg.build_profile, BuildProfile::Fast) {
                max_rounds.min(2 + stage as u32)
            } else {
                max_rounds
            };
            let fail_bucket_limit = fast_fail_bucket_limit(cfg.build_profile, buckets);

            let mut h1 = vec![0u64; n];
            let mut h2 = vec![0u64; n];
            let mut h3 = vec![0u64; n];
            let mut bucket_idx = vec![0usize; n];
            let mut counts = vec![0u32; buckets];
            let mut offsets = vec![0usize; buckets + 1];
            let mut cur = vec![0usize; buckets + 1];
            let mut items = vec![0u32; n];

            for round in 0..=stage_rounds {
                let salt = mix64(
                    cfg.seed
                        ^ (stage as u64).wrapping_mul(0xD6E8_FD9D_50E9_4A4D)
                        ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );

                let s1 = mix64(salt ^ 0x9E37_79B9_7F4A_7C15);
                let s2 = mix64(salt ^ 0xA24B_1F6F_DA39_2B31);
                let s3 = mix64(salt ^ 0xE703_7ED1_A0B4_28DB);
                crate::simd_hash::hash_u64(keys, s1, &mut h1);
                crate::simd_hash::hash_u64(keys, s2, &mut h2);
                crate::simd_hash::hash_u64(keys, s3, &mut h3);
                for v in &mut h3 {
                    *v |= 1;
                }

                for i in 0..n {
                    bucket_idx[i] = fast_reduce64(h1[i], buckets);
                }

                counts.fill(0);
                if cfg.enable_parallel_build && buckets >= 1024 && n >= (1 << 15) {
                    #[cfg(feature = "parallel")]
                    {
                        let shard_count = core_shard_count(cfg, buckets, n);
                        let chunk = n.div_ceil(shard_count);
                        let local_counts: Vec<Vec<u32>> = (0..shard_count)
                            .into_par_iter()
                            .map(|shard| {
                                let start = shard * chunk;
                                let end = (start + chunk).min(n);
                                let mut local = vec![0u32; buckets];
                                for i in start..end {
                                    local[bucket_idx[i]] += 1;
                                }
                                local
                            })
                            .collect();
                        for lc in local_counts {
                            for (i, &v) in lc.iter().enumerate() {
                                counts[i] += v;
                            }
                        }
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for &b in &bucket_idx {
                            counts[b] += 1;
                        }
                    }
                } else {
                    for &b in &bucket_idx {
                        counts[b] += 1;
                    }
                }

                offsets[0] = 0;
                for i in 0..buckets {
                    offsets[i + 1] = offsets[i] + counts[i] as usize;
                }
                cur.copy_from_slice(&offsets);
                for i in 0..n {
                    let b = bucket_idx[i];
                    let pos = cur[b];
                    items[pos] = i as u32;
                    cur[b] += 1;
                }

                let order = build_bucket_order_by_count(&counts);

                let mut occupied = vec![false; n];
                let mut seen_epoch = vec![0u32; n];
                let mut pilots = vec![0u32; buckets];
                let mut epoch = 1u32;
                let mut tmp = vec![0usize; 32];
                let mut ok_all = true;
                let mut failed_buckets = 0usize;

                for &b_u32 in &order {
                    let b = b_u32 as usize;
                    let start = offsets[b];
                    let end = offsets[b + 1];
                    let len = end - start;
                    if len == 0 {
                        continue;
                    }
                    if tmp.len() < len {
                        tmp.resize(len, 0);
                    }

                    let max_attempts = pilot_budget.max((len as u32) * 24);
                    let mut placed = false;
                    for attempt in 0..max_attempts {
                        epoch = epoch.wrapping_add(1);
                        if epoch == 0 {
                            seen_epoch.fill(0);
                            epoch = 1;
                        }
                        let pilot = mix64(
                            salt ^ ((b_u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                                ^ ((attempt as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
                        ) as u32;

                        let mut valid = true;
                        for i in 0..len {
                            let idx = items[start + i] as usize;
                            let slot = fast_reduce64(
                                mix64(
                                    h2[idx]
                                        ^ h3[idx].wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                        ^ (pilot as u64).wrapping_mul(0xA24B_1F6F_DA39_2B31),
                                ),
                                n,
                            );
                            if occupied[slot] || seen_epoch[slot] == epoch {
                                valid = false;
                                break;
                            }
                            seen_epoch[slot] = epoch;
                            tmp[i] = slot;
                        }

                        if !valid {
                            continue;
                        }
                        for &s in &tmp[..len] {
                            occupied[s] = true;
                        }
                        pilots[b] = pilot;
                        placed = true;
                        break;
                    }

                    if !placed {
                        failed_buckets += 1;
                        ok_all = false;
                        if round < cfg.fast_fail_rounds && failed_buckets >= fail_bucket_limit {
                            break;
                        }
                        break;
                    }
                }

                if ok_all {
                    return Some(Self {
                        n: n as u32,
                        m: buckets as u32,
                        seed: salt,
                        hash_s1: s1,
                        hash_s2: s2,
                        hash_s3: s3,
                        pilots,
                    });
                }
            }
        }

        None
    }

    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        if self.n == 0 {
            return None;
        }
        let h1 = crate::simd_hash::hash_u64_one(key, self.hash_s1);
        let h2 = crate::simd_hash::hash_u64_one(key, self.hash_s2);
        let h3 = crate::simd_hash::hash_u64_one(key, self.hash_s3) | 1;
        let b = fast_reduce64(h1, self.m as usize);
        let pilot = unsafe { *self.pilots.get_unchecked(b) } as u64;
        let slot = fast_reduce64(
            mix64(
                h2 ^ h3.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ pilot.wrapping_mul(0xA24B_1F6F_DA39_2B31),
            ),
            self.n as usize,
        ) as u32;
        Some(slot)
    }

    fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.pilots.len() * std::mem::size_of::<u32>()
    }
}

#[derive(Debug, Clone)]
struct ChdCore {
    n: u32,
    m: u32,
    seed: u64,
    hash_s1: u64,
    hash_s2: u64,
    hash_s3: u64,
    slot_starts: Vec<u32>,
    disps: Vec<u32>,
}

impl ChdCore {
    fn build(keys: &[u64], cfg: &BuildConfig, gamma: f64, attempt_budget: u32) -> Option<Self> {
        if keys.is_empty() {
            return None;
        }
        let n = keys.len();
        let fast = matches!(cfg.build_profile, BuildProfile::Fast);
        let gamma_schedule = chd_gamma_schedule(fast, gamma);
        let rounds = if fast {
            cfg.rehash_limit.min(2).max(1)
        } else {
            cfg.rehash_limit.max(1)
        };

        for (stage, stage_gamma) in gamma_schedule.into_iter().enumerate() {
            let buckets = ((n as f64 / stage_gamma.max(0.55)).ceil() as usize).max(1);
            let use_sharded = std::env::var_os("KIRA_CHD_SHARDED").is_some();
            let shard_count = if use_sharded {
                chd_shard_count(cfg, buckets, n)
            } else {
                1
            };
            let slot_starts = chd_slot_starts(n, shard_count);

            let mut h1 = vec![0u64; n];
            let mut h2 = vec![0u64; n];
            let mut h3 = vec![0u64; n];
            let mut bucket_idx = vec![0u32; n];
            let mut counts = vec![0u32; buckets];
            let mut offsets = vec![0u32; buckets + 1];
            let mut cur = vec![0u32; buckets + 1];
            let mut items = vec![0u32; n];

            for round in 0..=rounds {
                let seed = mix64(
                    cfg.seed
                        ^ 0xC84D_9D89_D6C1_1A7F
                        ^ (stage as u64).wrapping_mul(0xD6E8_FD9D_50E9_4A4D)
                        ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let s1 = mix64(seed ^ 0x9E37_79B9_7F4A_7C15);
                let s2 = mix64(seed ^ 0xA24B_1F6F_DA39_2B31);
                let s3 = mix64(seed ^ 0xE703_7ED1_A0B4_28DB);

                crate::simd_hash::hash_u64(keys, s1, &mut h1);
                crate::simd_hash::hash_u64(keys, s2, &mut h2);
                crate::simd_hash::hash_u64(keys, s3, &mut h3);
                for i in 0..n {
                    unsafe {
                        *h3.get_unchecked_mut(i) = *h3.get_unchecked(i) | 1;
                    }
                }

                for i in 0..n {
                    bucket_idx[i] = fast_reduce64(h1[i], buckets) as u32;
                }

                counts.fill(0);
                if cfg.enable_parallel_build && buckets >= 1024 && n >= (1 << 15) {
                    #[cfg(feature = "parallel")]
                    {
                        let count_shards = core_shard_count(cfg, buckets, n);
                        let chunk = n.div_ceil(count_shards);
                        let local_counts: Vec<Vec<u32>> = (0..count_shards)
                            .into_par_iter()
                            .map(|shard| {
                                let start = shard * chunk;
                                let end = (start + chunk).min(n);
                                let mut local = vec![0u32; buckets];
                                for i in start..end {
                                    local[unsafe { *bucket_idx.get_unchecked(i) as usize }] += 1;
                                }
                                local
                            })
                            .collect();
                        for lc in local_counts {
                            for (i, &v) in lc.iter().enumerate() {
                                counts[i] += v;
                            }
                        }
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for &b in &bucket_idx {
                            counts[b as usize] += 1;
                        }
                    }
                } else {
                    for &b in &bucket_idx {
                        counts[b as usize] += 1;
                    }
                }

                offsets[0] = 0;
                for i in 0..buckets {
                    offsets[i + 1] = offsets[i] + counts[i];
                }
                cur.copy_from_slice(&offsets);
                for i in 0..n {
                    let b = unsafe { *bucket_idx.get_unchecked(i) as usize };
                    let pos = unsafe { *cur.get_unchecked(b) as usize };
                    unsafe { *items.get_unchecked_mut(pos) = i as u32 };
                    unsafe { *cur.get_unchecked_mut(b) += 1 };
                }

                let order = build_bucket_order_by_count(&counts);
                if shard_count > 1 {
                    let mut shard_orders = vec![Vec::<u32>::new(); shard_count];
                    for &b in &order {
                        let sid = ((b as usize) * shard_count) / buckets;
                        shard_orders[sid.min(shard_count - 1)].push(b);
                    }

                    #[cfg(feature = "parallel")]
                    let shard_results = if cfg.enable_parallel_build {
                        shard_orders
                            .par_iter()
                            .enumerate()
                            .map(|(sid, shard_order)| {
                                solve_chd_shard(
                                    shard_order,
                                    &offsets,
                                    &items,
                                    &h2,
                                    &h3,
                                    seed,
                                    stage,
                                    attempt_budget,
                                    fast,
                                    (slot_starts[sid + 1] - slot_starts[sid]) as usize,
                                    cfg.build_profile,
                                    cfg.fast_fail_rounds,
                                    round,
                                    sid,
                                )
                            })
                            .collect::<Vec<Option<Vec<(u32, u32)>>>>()
                    } else {
                        shard_orders
                            .iter()
                            .enumerate()
                            .map(|(sid, shard_order)| {
                                solve_chd_shard(
                                    shard_order,
                                    &offsets,
                                    &items,
                                    &h2,
                                    &h3,
                                    seed,
                                    stage,
                                    attempt_budget,
                                    fast,
                                    (slot_starts[sid + 1] - slot_starts[sid]) as usize,
                                    cfg.build_profile,
                                    cfg.fast_fail_rounds,
                                    round,
                                    sid,
                                )
                            })
                            .collect::<Vec<Option<Vec<(u32, u32)>>>>()
                    };
                    #[cfg(not(feature = "parallel"))]
                    let shard_results = shard_orders
                        .iter()
                        .enumerate()
                        .map(|(sid, shard_order)| {
                            solve_chd_shard(
                                shard_order,
                                &offsets,
                                &items,
                                &h2,
                                &h3,
                                seed,
                                stage,
                                attempt_budget,
                                fast,
                                (slot_starts[sid + 1] - slot_starts[sid]) as usize,
                                cfg.build_profile,
                                cfg.fast_fail_rounds,
                                round,
                                sid,
                            )
                        })
                        .collect::<Vec<Option<Vec<(u32, u32)>>>>();

                    let mut disps = vec![0u32; buckets];
                    let mut ok_all = true;
                    for shard in shard_results {
                        if let Some(assignments) = shard {
                            for (b, d) in assignments {
                                unsafe { *disps.get_unchecked_mut(b as usize) = d };
                            }
                        } else {
                            ok_all = false;
                            break;
                        }
                    }
                    if ok_all {
                        return Some(Self {
                            n: n as u32,
                            m: buckets as u32,
                            seed,
                            hash_s1: s1,
                            hash_s2: s2,
                            hash_s3: s3,
                            slot_starts: slot_starts.clone(),
                            disps,
                        });
                    }
                }

                if let Some(disps) = solve_chd_global(
                    &order,
                    &offsets,
                    &items,
                    &h2,
                    &h3,
                    n,
                    seed,
                    stage,
                    attempt_budget,
                    fast,
                    cfg.build_profile,
                    cfg.fast_fail_rounds,
                    round,
                ) {
                    return Some(Self {
                        n: n as u32,
                        m: buckets as u32,
                        seed,
                        hash_s1: s1,
                        hash_s2: s2,
                        hash_s3: s3,
                        slot_starts: vec![0u32, n as u32],
                        disps,
                    });
                }
            }
        }

        None
    }

    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        if self.n == 0 {
            return None;
        }
        let h1 = crate::simd_hash::hash_u64_one(key, self.hash_s1);
        let h2 = crate::simd_hash::hash_u64_one(key, self.hash_s2);
        let h3 = crate::simd_hash::hash_u64_one(key, self.hash_s3) | 1;
        let b = fast_reduce64(h1, self.m as usize);
        let shard_count = self.slot_starts.len() - 1;
        let sid = (b * shard_count) / (self.m as usize);
        let slot_start = unsafe { *self.slot_starts.get_unchecked(sid) as usize };
        let slot_len = (unsafe {
            *self.slot_starts.get_unchecked(sid + 1) - *self.slot_starts.get_unchecked(sid)
        }) as usize;
        if slot_len == 0 {
            return None;
        }
        let disp = unsafe { *self.disps.get_unchecked(b) } as u64;
        let local = chd_slot(h2, h3, disp as u32, slot_len);
        Some((slot_start + local) as u32)
    }

    fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.disps.len() * std::mem::size_of::<u32>()
            + self.slot_starts.len() * std::mem::size_of::<u32>()
    }
}

fn solve_chd_shard(
    shard_order: &[u32],
    offsets: &[u32],
    items: &[u32],
    h2: &[u64],
    h3: &[u64],
    seed: u64,
    stage: usize,
    attempt_budget: u32,
    fast: bool,
    slot_len: usize,
    profile: BuildProfile,
    fast_fail_rounds: u32,
    round: u32,
    shard_id: usize,
) -> Option<Vec<(u32, u32)>> {
    if shard_order.is_empty() {
        return Some(Vec::new());
    }
    if slot_len == 0 {
        return None;
    }

    let fail_limit = fast_fail_bucket_limit(profile, shard_order.len());
    let mut occupied = vec![0u64; slot_len.div_ceil(64)];
    let mut seen_epoch = vec![0u32; slot_len];
    let mut epoch = 1u32;
    let mut tmp = vec![0u32; 32];
    let mut failed = 0usize;
    let mut out = Vec::with_capacity(shard_order.len());

    for &b_u32 in shard_order {
        let b = b_u32 as usize;
        let start = unsafe { *offsets.get_unchecked(b) as usize };
        let end = unsafe { *offsets.get_unchecked(b + 1) as usize };
        let len = end - start;
        if len == 0 {
            out.push((b_u32, 0));
            continue;
        }
        if tmp.len() < len {
            tmp.resize(len, 0);
        }
        let mut cap =
            chd_attempt_cap(len, fast).min(attempt_budget.max((len as u32).saturating_mul(8)));
        if cap == 0 {
            cap = 1;
        }
        cap = cap.saturating_add((stage as u32).saturating_mul(256));
        let mut placed = false;
        let mut disp_state = mix64(
            seed ^ (b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (shard_id as u64).wrapping_mul(0xA24B_1F6F_DA39_2B31),
        );

        for attempt in 0..cap {
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                seen_epoch.fill(0);
                epoch = 1;
            }
            let disp = (disp_state as u32).wrapping_add(attempt.wrapping_mul(0x9E37_79B9));
            disp_state = xorshift64(disp_state ^ 0xBF58_476D_1CE4_E5B9);

            let mut valid = true;
            for i in 0..len {
                let idx = unsafe { *items.get_unchecked(start + i) as usize };
                let slot = chd_slot(
                    unsafe { *h2.get_unchecked(idx) },
                    unsafe { *h3.get_unchecked(idx) },
                    disp,
                    slot_len,
                );
                if occupied_test(&occupied, slot)
                    || unsafe { *seen_epoch.get_unchecked(slot) } == epoch
                {
                    valid = false;
                    break;
                }
                unsafe { *seen_epoch.get_unchecked_mut(slot) = epoch };
                tmp[i] = slot as u32;
            }
            if !valid {
                continue;
            }
            for &s in &tmp[..len] {
                occupied_set(&mut occupied, s as usize);
            }
            placed = true;
            out.push((b_u32, disp));
            break;
        }

        if !placed {
            failed += 1;
            if round < fast_fail_rounds && failed >= fail_limit {
                return None;
            }
            return None;
        }
    }

    Some(out)
}

fn solve_chd_global(
    order: &[u32],
    offsets: &[u32],
    items: &[u32],
    h2: &[u64],
    h3: &[u64],
    n: usize,
    seed: u64,
    stage: usize,
    attempt_budget: u32,
    fast: bool,
    profile: BuildProfile,
    fast_fail_rounds: u32,
    round: u32,
) -> Option<Vec<u32>> {
    let mut occupied = vec![0u64; n.div_ceil(64)];
    let mut seen_epoch = vec![0u32; n];
    let mut epoch = 1u32;
    let mut tmp = vec![0u32; 32];
    let mut disps = vec![0u32; offsets.len().saturating_sub(1)];
    let fail_limit = fast_fail_bucket_limit(profile, disps.len());
    let mut failed = 0usize;

    for &b_u32 in order {
        let b = b_u32 as usize;
        let start = unsafe { *offsets.get_unchecked(b) as usize };
        let end = unsafe { *offsets.get_unchecked(b + 1) as usize };
        let len = end - start;
        if len == 0 {
            continue;
        }
        if tmp.len() < len {
            tmp.resize(len, 0);
        }
        let mut cap =
            chd_attempt_cap(len, fast).min(attempt_budget.max((len as u32).saturating_mul(8)));
        if cap == 0 {
            cap = 1;
        }
        cap = cap.saturating_add((stage as u32).saturating_mul(256));
        let mut placed = false;
        let mut disp_state = mix64(seed ^ (b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

        for attempt in 0..cap {
            epoch = epoch.wrapping_add(1);
            if epoch == 0 {
                seen_epoch.fill(0);
                epoch = 1;
            }
            let disp = (disp_state as u32).wrapping_add(attempt.wrapping_mul(0x9E37_79B9));
            disp_state = xorshift64(disp_state ^ 0xBF58_476D_1CE4_E5B9);

            let mut valid = true;
            for i in 0..len {
                let idx = unsafe { *items.get_unchecked(start + i) as usize };
                let slot = chd_slot(
                    unsafe { *h2.get_unchecked(idx) },
                    unsafe { *h3.get_unchecked(idx) },
                    disp,
                    n,
                );
                if occupied_test(&occupied, slot)
                    || unsafe { *seen_epoch.get_unchecked(slot) } == epoch
                {
                    valid = false;
                    break;
                }
                unsafe { *seen_epoch.get_unchecked_mut(slot) = epoch };
                tmp[i] = slot as u32;
            }
            if !valid {
                continue;
            }
            for &s in &tmp[..len] {
                occupied_set(&mut occupied, s as usize);
            }
            disps[b] = disp;
            placed = true;
            break;
        }

        if !placed {
            failed += 1;
            if round < fast_fail_rounds && failed >= fail_limit {
                return None;
            }
            return None;
        }
    }
    Some(disps)
}

#[derive(Debug, Clone)]
pub struct PTHashBackend {
    storage: BackendStorage,
}

impl MphBackend for PTHashBackend {
    fn build(keys: &[u64], config: &BuildConfig) -> Self {
        let gamma = config.gamma.max(1.0);
        let core = CoreMph::build(keys, config, gamma, config.max_pilot_attempts)
            .or_else(|| CoreMph::build(keys, config, 0.8, config.max_pilot_attempts * 2))
            .map(BackendStorage::Core)
            .unwrap_or_else(|| BackendStorage::Map(build_fallback_map(keys)));
        Self { storage: core }
    }

    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        self.storage.lookup(key)
    }

    fn memory_usage_bytes(&self) -> usize {
        self.storage.memory_usage_bytes()
    }
}

#[derive(Debug, Clone)]
enum PtrHash2025Storage {
    Ptr(crate::ptrhash::Mphf),
    Map(HashMap<u64, u32>),
}

#[derive(Debug, Clone)]
pub struct PtrHash2025Backend {
    storage: PtrHash2025Storage,
}

impl MphBackend for PtrHash2025Backend {
    fn build(keys: &[u64], config: &BuildConfig) -> Self {
        let fast = matches!(config.build_profile, BuildProfile::Fast);
        let mph_cfg = crate::ptrhash::BuildConfig {
            gamma: if fast {
                (config.gamma * 0.85).max(0.7)
            } else {
                config.gamma.max(0.8)
            },
            rehash_limit: if fast {
                1
            } else {
                config.rehash_limit.min(6).max(2)
            },
            salt: config.seed,
        };

        let build = crate::ptrhash::Builder::new()
            .with_config(mph_cfg)
            .build_unique_u64_ref(keys);

        let storage = match build {
            Ok(mph) => PtrHash2025Storage::Ptr(mph),
            Err(_) => PtrHash2025Storage::Map(build_fallback_map(keys)),
        };
        Self { storage }
    }

    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        match &self.storage {
            PtrHash2025Storage::Ptr(mph) => Some(mph.index_u64(key) as u32),
            PtrHash2025Storage::Map(map) => map.get(&key).copied(),
        }
    }

    fn memory_usage_bytes(&self) -> usize {
        match &self.storage {
            PtrHash2025Storage::Ptr(mph) => {
                std::mem::size_of::<crate::ptrhash::Mphf>()
                    + mph.g.len() * std::mem::size_of::<u32>()
            }
            PtrHash2025Storage::Map(map) => {
                std::mem::size_of::<HashMap<u64, u32>>()
                    + map.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CHDBackend {
    inner: PTHashBackend,
}

impl MphBackend for CHDBackend {
    fn build(keys: &[u64], config: &BuildConfig) -> Self {
        Self {
            inner: PTHashBackend::build(keys, config),
        }
    }

    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        self.inner.lookup(key)
    }

    fn memory_usage_bytes(&self) -> usize {
        self.inner.memory_usage_bytes()
    }
}

#[derive(Debug, Clone)]
pub struct RecSplitBackend {
    storage: BackendStorage,
}

impl MphBackend for RecSplitBackend {
    fn build(keys: &[u64], config: &BuildConfig) -> Self {
        let gamma = (config.gamma * 1.15).max(1.1);
        let budget = config.max_pilot_attempts / 2;
        let core = CoreMph::build(keys, config, gamma, budget.max(8_192))
            .or_else(|| CoreMph::build(keys, config, 1.0, config.max_pilot_attempts))
            .or_else(|| CoreMph::build(keys, config, 0.8, config.max_pilot_attempts * 2))
            .map(BackendStorage::Core)
            .unwrap_or_else(|| BackendStorage::Map(build_fallback_map(keys)));
        Self { storage: core }
    }

    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        self.storage.lookup(key)
    }

    fn memory_usage_bytes(&self) -> usize {
        self.storage.memory_usage_bytes()
    }
}

#[cfg(feature = "bbhash")]
#[derive(Debug, Clone)]
pub struct BBHashBackend {
    storage: BackendStorage,
}

#[cfg(feature = "bbhash")]
impl MphBackend for BBHashBackend {
    fn build(keys: &[u64], config: &BuildConfig) -> Self {
        let gamma = (config.gamma * 1.05).max(1.0);
        let core = CoreMph::build(keys, config, gamma, config.max_pilot_attempts)
            .map(BackendStorage::Core)
            .unwrap_or_else(|| BackendStorage::Map(build_fallback_map(keys)));
        Self { storage: core }
    }

    fn lookup(&self, key: u64) -> Option<u32> {
        self.storage.lookup(key)
    }

    fn memory_usage_bytes(&self) -> usize {
        self.storage.memory_usage_bytes()
    }
}

#[derive(Debug, Clone)]
enum BackendStorage {
    Core(CoreMph),
    Chd(ChdCore),
    PtrHash25(crate::ptrhash::Mphf),
    Map(HashMap<u64, u32>),
}

impl BackendStorage {
    #[inline]
    fn lookup(&self, key: u64) -> Option<u32> {
        match self {
            Self::Core(core) => core.lookup(key),
            Self::Chd(core) => core.lookup(key),
            Self::PtrHash25(mph) => Some(mph.index_u64(key) as u32),
            Self::Map(map) => map.get(&key).copied(),
        }
    }

    fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::Core(core) => core.memory_usage_bytes(),
            Self::Chd(core) => core.memory_usage_bytes(),
            Self::PtrHash25(mph) => {
                std::mem::size_of::<crate::ptrhash::Mphf>()
                    + mph.g.len() * std::mem::size_of::<u32>()
            }
            Self::Map(map) => {
                std::mem::size_of::<HashMap<u64, u32>>()
                    + map.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>())
            }
        }
    }
}

#[derive(Debug)]
pub enum BackendDispatch {
    PTHash(PTHashBackend),
    PtrHash2025(PtrHash2025Backend),
    CHD(CHDBackend),
    RecSplit(RecSplitBackend),
    #[cfg(feature = "bbhash")]
    BBHash(BBHashBackend),
}

impl BackendDispatch {
    pub fn kind(&self) -> BackendKind {
        match self {
            Self::PTHash(_) => BackendKind::PTHash,
            Self::PtrHash2025(_) => BackendKind::PtrHash2025,
            Self::CHD(_) => BackendKind::CHD,
            Self::RecSplit(_) => BackendKind::RecSplit,
            #[cfg(feature = "bbhash")]
            Self::BBHash(_) => BackendKind::BBHash,
        }
    }

    #[inline]
    pub fn lookup(&self, key: u64) -> Option<u32> {
        match self {
            Self::PTHash(b) => b.lookup(key),
            Self::PtrHash2025(b) => b.lookup(key),
            Self::CHD(b) => b.lookup(key),
            Self::RecSplit(b) => b.lookup(key),
            #[cfg(feature = "bbhash")]
            Self::BBHash(b) => b.lookup(key),
        }
    }

    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::PTHash(b) => b.memory_usage_bytes(),
            Self::PtrHash2025(b) => b.memory_usage_bytes(),
            Self::CHD(b) => b.memory_usage_bytes(),
            Self::RecSplit(b) => b.memory_usage_bytes(),
            #[cfg(feature = "bbhash")]
            Self::BBHash(b) => b.memory_usage_bytes(),
        }
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        match self {
            Self::PTHash(b) => {
                out.push(0);
                write_storage(&b.storage, out);
            }
            Self::PtrHash2025(b) => {
                out.push(4);
                write_ptrhash2025_storage(&b.storage, out);
            }
            Self::CHD(b) => {
                out.push(1);
                write_storage(&b.inner.storage, out);
            }
            Self::RecSplit(b) => {
                out.push(2);
                write_storage(&b.storage, out);
            }
            #[cfg(feature = "bbhash")]
            Self::BBHash(b) => {
                out.push(3);
                write_storage(&b.storage, out);
            }
        }
    }

    pub fn read_from(buf: &[u8], pos: &mut usize) -> Option<Self> {
        let tag = read_u8(buf, pos)?;
        match tag {
            0 => {
                let storage = read_storage(buf, pos)?;
                Some(Self::PTHash(PTHashBackend { storage }))
            }
            1 => {
                let storage = read_storage(buf, pos)?;
                Some(Self::CHD(CHDBackend {
                    inner: PTHashBackend { storage },
                }))
            }
            2 => {
                let storage = read_storage(buf, pos)?;
                Some(Self::RecSplit(RecSplitBackend { storage }))
            }
            4 => {
                let storage = read_ptrhash2025_storage(buf, pos)?;
                Some(Self::PtrHash2025(PtrHash2025Backend { storage }))
            }
            #[cfg(feature = "bbhash")]
            3 => {
                let storage = read_storage(buf, pos)?;
                Some(Self::BBHash(BBHashBackend { storage }))
            }
            _ => None,
        }
    }
}

pub fn build_dispatch(keys: &[u64], cfg: &BuildConfig) -> BackendDispatch {
    match cfg.backend {
        BackendKind::PTHash => BackendDispatch::PTHash(PTHashBackend::build(keys, cfg)),
        BackendKind::PtrHash2025 => {
            BackendDispatch::PtrHash2025(PtrHash2025Backend::build(keys, cfg))
        }
        BackendKind::CHD => BackendDispatch::CHD(CHDBackend::build(keys, cfg)),
        BackendKind::RecSplit => BackendDispatch::RecSplit(RecSplitBackend::build(keys, cfg)),
        #[cfg(feature = "bbhash")]
        BackendKind::BBHash => BackendDispatch::BBHash(BBHashBackend::build(keys, cfg)),
    }
}

fn write_core(core: &CoreMph, out: &mut Vec<u8>) {
    out.extend_from_slice(&core.n.to_le_bytes());
    out.extend_from_slice(&core.m.to_le_bytes());
    out.extend_from_slice(&core.seed.to_le_bytes());
    out.extend_from_slice(&(core.pilots.len() as u64).to_le_bytes());
    for &p in &core.pilots {
        out.extend_from_slice(&p.to_le_bytes());
    }
}

fn read_core(buf: &[u8], pos: &mut usize) -> Option<CoreMph> {
    let n = read_u32(buf, pos)?;
    let m = read_u32(buf, pos)?;
    let seed = read_u64(buf, pos)?;
    let len = read_u64(buf, pos)? as usize;
    let mut pilots = Vec::with_capacity(len);
    for _ in 0..len {
        pilots.push(read_u32(buf, pos)?);
    }
    Some(CoreMph {
        n,
        m,
        seed,
        hash_s1: mix64(seed ^ 0x9E37_79B9_7F4A_7C15),
        hash_s2: mix64(seed ^ 0xA24B_1F6F_DA39_2B31),
        hash_s3: mix64(seed ^ 0xE703_7ED1_A0B4_28DB),
        pilots,
    })
}

fn write_chd(core: &ChdCore, out: &mut Vec<u8>) {
    out.extend_from_slice(&core.n.to_le_bytes());
    out.extend_from_slice(&core.m.to_le_bytes());
    out.extend_from_slice(&core.seed.to_le_bytes());
    out.extend_from_slice(&(core.slot_starts.len() as u64).to_le_bytes());
    for &v in &core.slot_starts {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(core.disps.len() as u64).to_le_bytes());
    for &p in &core.disps {
        out.extend_from_slice(&p.to_le_bytes());
    }
}

fn read_chd(buf: &[u8], pos: &mut usize) -> Option<ChdCore> {
    let n = read_u32(buf, pos)?;
    let m = read_u32(buf, pos)?;
    let seed = read_u64(buf, pos)?;
    let slot_len = read_u64(buf, pos)? as usize;
    let mut slot_starts = Vec::with_capacity(slot_len);
    for _ in 0..slot_len {
        slot_starts.push(read_u32(buf, pos)?);
    }
    if slot_starts.len() < 2 {
        slot_starts = vec![0u32, n];
    }
    let len = read_u64(buf, pos)? as usize;
    let mut disps = Vec::with_capacity(len);
    for _ in 0..len {
        disps.push(read_u32(buf, pos)?);
    }
    Some(ChdCore {
        n,
        m,
        seed,
        hash_s1: mix64(seed ^ 0x9E37_79B9_7F4A_7C15),
        hash_s2: mix64(seed ^ 0xA24B_1F6F_DA39_2B31),
        hash_s3: mix64(seed ^ 0xE703_7ED1_A0B4_28DB),
        slot_starts,
        disps,
    })
}

fn write_storage(storage: &BackendStorage, out: &mut Vec<u8>) {
    match storage {
        BackendStorage::Core(core) => {
            out.push(0);
            write_core(core, out);
        }
        BackendStorage::Chd(core) => {
            out.push(1);
            write_chd(core, out);
        }
        BackendStorage::PtrHash25(mph) => {
            out.push(3);
            out.extend_from_slice(&mph.n.to_le_bytes());
            out.extend_from_slice(&mph.m.to_le_bytes());
            out.extend_from_slice(&mph.salt.to_le_bytes());
            out.extend_from_slice(&(mph.g.len() as u64).to_le_bytes());
            for &v in &mph.g {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        BackendStorage::Map(map) => {
            out.push(2);
            out.extend_from_slice(&(map.len() as u64).to_le_bytes());
            for (k, v) in map {
                out.extend_from_slice(&k.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

fn read_storage(buf: &[u8], pos: &mut usize) -> Option<BackendStorage> {
    let tag = read_u8(buf, pos)?;
    match tag {
        0 => read_core(buf, pos).map(BackendStorage::Core),
        1 => read_chd(buf, pos).map(BackendStorage::Chd),
        2 => {
            let len = read_u64(buf, pos)? as usize;
            let mut map = HashMap::with_capacity(len * 2);
            for _ in 0..len {
                let k = read_u64(buf, pos)?;
                let v = read_u32(buf, pos)?;
                map.insert(k, v);
            }
            Some(BackendStorage::Map(map))
        }
        3 => {
            let n = read_u64(buf, pos)?;
            let m = read_u32(buf, pos)?;
            let salt = read_u64(buf, pos)?;
            let len = read_u64(buf, pos)? as usize;
            let mut g = Vec::with_capacity(len);
            for _ in 0..len {
                g.push(read_u32(buf, pos)?);
            }
            Some(BackendStorage::PtrHash25(crate::ptrhash::Mphf {
                n,
                m,
                salt,
                g,
            }))
        }
        _ => None,
    }
}

fn write_ptrhash2025_storage(storage: &PtrHash2025Storage, out: &mut Vec<u8>) {
    match storage {
        PtrHash2025Storage::Ptr(mph) => {
            out.push(0);
            out.extend_from_slice(&mph.n.to_le_bytes());
            out.extend_from_slice(&mph.m.to_le_bytes());
            out.extend_from_slice(&mph.salt.to_le_bytes());
            out.extend_from_slice(&(mph.g.len() as u64).to_le_bytes());
            for &v in &mph.g {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        PtrHash2025Storage::Map(map) => {
            out.push(1);
            out.extend_from_slice(&(map.len() as u64).to_le_bytes());
            for (k, v) in map {
                out.extend_from_slice(&k.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

fn read_ptrhash2025_storage(buf: &[u8], pos: &mut usize) -> Option<PtrHash2025Storage> {
    let tag = read_u8(buf, pos)?;
    match tag {
        0 => {
            let n = read_u64(buf, pos)?;
            let m = read_u32(buf, pos)?;
            let salt = read_u64(buf, pos)?;
            let len = read_u64(buf, pos)? as usize;
            let mut g = Vec::with_capacity(len);
            for _ in 0..len {
                g.push(read_u32(buf, pos)?);
            }
            Some(PtrHash2025Storage::Ptr(crate::ptrhash::Mphf {
                n,
                m,
                salt,
                g,
            }))
        }
        1 => {
            let len = read_u64(buf, pos)? as usize;
            let mut map = HashMap::with_capacity(len * 2);
            for _ in 0..len {
                let k = read_u64(buf, pos)?;
                let v = read_u32(buf, pos)?;
                map.insert(k, v);
            }
            Some(PtrHash2025Storage::Map(map))
        }
        _ => None,
    }
}

fn build_fallback_map(keys: &[u64]) -> HashMap<u64, u32> {
    let mut map = HashMap::with_capacity(keys.len() * 2);
    for (i, &k) in keys.iter().enumerate() {
        map.insert(k, i as u32);
    }
    map
}

fn read_u8(buf: &[u8], pos: &mut usize) -> Option<u8> {
    if *pos + 1 > buf.len() {
        return None;
    }
    let v = buf[*pos];
    *pos += 1;
    Some(v)
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Option<u32> {
    if *pos + 4 > buf.len() {
        return None;
    }
    let mut a = [0u8; 4];
    a.copy_from_slice(&buf[*pos..*pos + 4]);
    *pos += 4;
    Some(u32::from_le_bytes(a))
}

fn read_u64(buf: &[u8], pos: &mut usize) -> Option<u64> {
    if *pos + 8 > buf.len() {
        return None;
    }
    let mut a = [0u8; 8];
    a.copy_from_slice(&buf[*pos..*pos + 8]);
    *pos += 8;
    Some(u64::from_le_bytes(a))
}

#[inline]
fn set_bit(bits: &mut [u64], idx: usize) {
    let w = idx >> 6;
    let b = idx & 63;
    bits[w] |= 1u64 << b;
}

#[inline]
fn test_bit(bits: &[u64], idx: usize) -> bool {
    let w = idx >> 6;
    let b = idx & 63;
    ((bits[w] >> b) & 1) != 0
}

#[inline]
fn fast_reduce64(hash: u64, n: usize) -> usize {
    ((hash as u128 * n as u128) >> 64) as usize
}

#[inline]
fn chd_slot(h2: u64, h3: u64, disp: u32, n: usize) -> usize {
    fast_reduce64(
        mix64(
            h2 ^ h3.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (disp as u64).wrapping_mul(0xA24B_1F6F_DA39_2B31),
        ),
        n,
    )
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn build_gamma_schedule(profile: BuildProfile, gamma: f64) -> [f64; 4] {
    let g = gamma.max(0.6);
    match profile {
        BuildProfile::Balanced => [g, (g * 0.9).max(0.7), (g * 0.8).max(0.65), 0.60],
        BuildProfile::Fast => [g, (g * 0.88).max(0.65), 0.62, 0.58],
    }
}

#[inline]
fn chd_gamma_schedule(fast: bool, gamma: f64) -> [f64; 4] {
    let g = gamma.max(0.58);
    if fast {
        [g, (g * 1.10).max(0.70), (g * 1.24).max(0.82), 1.0]
    } else {
        [g, (g * 1.08).max(0.72), (g * 1.20).max(0.88), 1.12]
    }
}

#[inline]
fn core_shard_count(cfg: &BuildConfig, buckets: usize, n: usize) -> usize {
    if buckets == 0 || n <= 1 {
        return 1;
    }
    if !cfg.enable_parallel_build {
        return 1;
    }
    #[cfg(feature = "parallel")]
    {
        if let Ok(par) = std::thread::available_parallelism() {
            let mut s = par.get().min(16).max(1);
            s = s.min(buckets).min(n).max(1);
            if matches!(cfg.build_profile, BuildProfile::Balanced) {
                return s.min(8).max(1);
            }
            return s;
        }
    }
    1
}

#[inline]
fn chd_shard_count(cfg: &BuildConfig, buckets: usize, n: usize) -> usize {
    if buckets == 0 || n <= 1 {
        return 1;
    }
    if !cfg.enable_parallel_build {
        return 1;
    }
    #[cfg(feature = "parallel")]
    {
        if let Ok(par) = std::thread::available_parallelism() {
            let mut s = par.get().min(12).max(1);
            if matches!(cfg.build_profile, BuildProfile::Balanced) {
                s = s.min(8);
            }
            s = s.min(buckets).min(n).max(1);
            return s;
        }
    }
    1
}

fn chd_slot_starts(n: usize, shards: usize) -> Vec<u32> {
    let s = shards.max(1).min(n.max(1));
    let mut starts = Vec::with_capacity(s + 1);
    starts.push(0u32);
    for sid in 1..s {
        let v = ((sid as u128 * n as u128) / s as u128) as u32;
        starts.push(v);
    }
    starts.push(n as u32);
    starts
}

#[inline]
fn fast_fail_bucket_limit(profile: BuildProfile, buckets: usize) -> usize {
    match profile {
        BuildProfile::Balanced => (buckets / 64).max(16),
        BuildProfile::Fast => (buckets / 128).max(8),
    }
}

fn build_bucket_order_by_count(counts: &[u32]) -> Vec<u32> {
    if counts.is_empty() {
        return Vec::new();
    }
    let max_len = counts.iter().copied().max().unwrap_or(0) as usize;
    let mut freq = vec![0usize; max_len + 1];
    for &c in counts {
        freq[c as usize] += 1;
    }

    let mut start = vec![0usize; max_len + 1];
    let mut acc = 0usize;
    for len in (0..=max_len).rev() {
        start[len] = acc;
        acc += freq[len];
    }

    let mut next = start;
    let mut order = vec![0u32; counts.len()];
    for (bucket, &c) in counts.iter().enumerate() {
        let pos = next[c as usize];
        order[pos] = bucket as u32;
        next[c as usize] = pos + 1;
    }
    order
}

#[inline]
fn chd_attempt_cap(len: usize, fast: bool) -> u32 {
    let base = match len {
        0..=2 => 256,
        3..=4 => 1_024,
        5..=8 => 4_096,
        9..=16 => 16_384,
        17..=32 => 65_536,
        _ => 131_072,
    };
    if fast { base } else { base.saturating_mul(2) }
}

#[inline]
fn occupied_test(bits: &[u64], idx: usize) -> bool {
    let word = unsafe { *bits.get_unchecked(idx >> 6) };
    let mask = 1u64 << (idx & 63);
    (word & mask) != 0
}

#[inline]
fn occupied_set(bits: &mut [u64], idx: usize) {
    let word = unsafe { bits.get_unchecked_mut(idx >> 6) };
    *word |= 1u64 << (idx & 63);
}

#[inline]
fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

pub fn prehash_unique_u64(keys: &[Vec<u8>], seed: u64) -> Option<(u64, Vec<u64>)> {
    for round in 0..64u64 {
        let s = seed ^ round.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut out = Vec::with_capacity(keys.len());
        let mut seen = HashSet::with_capacity(keys.len() * 2);
        let mut ok = true;
        for k in keys {
            let h = wyhash::wyhash(k, s);
            if !seen.insert(h) {
                ok = false;
                break;
            }
            out.push(h);
        }
        if ok {
            return Some((s, out));
        }
    }
    None
}

pub fn prehash_u64_arena(
    bytes: &[u8],
    offsets: &[u32],
    seed: u64,
    verify_uniqueness: bool,
) -> Option<(u64, Vec<u64>)> {
    if offsets.len() < 2 {
        return None;
    }
    let key_count = offsets.len() - 1;
    let rounds: u64 = if verify_uniqueness { 64 } else { 1 };
    for round in 0..rounds {
        let s = seed ^ round.wrapping_mul(0x9E37_79B9_7F4A_7C15);

        #[cfg(feature = "parallel")]
        let hashes: Vec<u64> = offsets
            .par_windows(2)
            .map(|w| {
                let start = w[0] as usize;
                let end = w[1] as usize;
                wyhash::wyhash(&bytes[start..end], s)
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let hashes: Vec<u64> = offsets
            .windows(2)
            .map(|w| {
                let start = w[0] as usize;
                let end = w[1] as usize;
                wyhash::wyhash(&bytes[start..end], s)
            })
            .collect();

        if !verify_uniqueness {
            return Some((s, hashes));
        }

        let mut out = Vec::with_capacity(key_count);
        let mut seen = HashSet::with_capacity(key_count * 2);
        let mut ok = true;
        for h in hashes.into_iter() {
            if !seen.insert(h) {
                ok = false;
                break;
            }
            out.push(h);
        }
        if ok {
            return Some((s, out));
        }
    }
    None
}

pub fn prehash_unique_u64_arena(
    bytes: &[u8],
    offsets: &[u32],
    seed: u64,
) -> Option<(u64, Vec<u64>)> {
    prehash_u64_arena(bytes, offsets, seed, true)
}
