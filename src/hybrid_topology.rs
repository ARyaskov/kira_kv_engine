//! Hybrid CPU topology detection.
//!
//! Returns the list of OS logical core IDs that correspond to **P-cores** (performance
//! cores) on hybrid CPUs like Intel Alder/Raptor Lake (12th/13th/14th gen) or Apple M-series.
//!
//! - On Windows we walk `GetLogicalProcessorInformationEx(RelationProcessorCore)` and rank
//!   cores by their EfficiencyClass.
//! - On Linux we read `/sys/devices/system/cpu/cpu*/cpu_capacity` (preferred) or
//!   `cpufreq/cpuinfo_max_freq` and keep the highest-tier cores.
//! - On other platforms we fall back to "all cores", matching prior behaviour.
//!
//! When the topology is homogeneous (no efficiency classes / equal frequencies) we
//! return all physical cores, deduplicated across SMT siblings.

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Topology {
    /// OS logical core IDs that belong to P-cores. Empty if detection failed.
    pub performance_cores: Vec<usize>,
    /// OS logical core IDs that belong to E-cores. Empty if the host is homogeneous.
    pub efficiency_cores: Vec<usize>,
    /// Whether the host is heterogeneous (hybrid).
    pub is_hybrid: bool,
}

impl Topology {
    pub fn detect() -> Self {
        #[cfg(target_os = "windows")]
        {
            if let Some(t) = windows::detect() {
                return t;
            }
        }
        #[cfg(target_os = "linux")]
        {
            if let Some(t) = linux::detect() {
                return t;
            }
        }
        Self::fallback()
    }

    fn fallback() -> Self {
        let n = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1);
        Self {
            performance_cores: (0..n).collect(),
            efficiency_cores: Vec::new(),
            is_hybrid: false,
        }
    }
}

/// Best-effort: returns the build-side preferred core list — P-cores if hybrid,
/// otherwise the first N distinct physical cores.
#[allow(dead_code)]
pub fn preferred_build_cores() -> Vec<usize> {
    let t = Topology::detect();
    if t.performance_cores.is_empty() {
        return (0..std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1))
            .collect();
    }
    t.performance_cores
}

/// Role-based core split for build pipeline.
// ----- Windows
#[cfg(target_os = "windows")]
mod windows {
    use super::Topology;
    use std::mem;

    // Minimal manual bindings to GetLogicalProcessorInformationEx to avoid pulling in winapi.
    #[repr(C)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }

    #[repr(C)]
    struct ProcessorRelationship {
        flags: u8,
        efficiency_class: u8,
        reserved: [u8; 20],
        group_count: u16,
        // Followed by group_count GroupAffinity entries.
        groups: [GroupAffinity; 1],
    }

    #[repr(C)]
    struct LogicalProcInfoEx {
        relationship: u32,
        size: u32,
        // The body is a union; we only read the ProcessorRelationship variant.
        processor: ProcessorRelationship,
    }

    const RELATION_PROCESSOR_CORE: u32 = 0;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship_type: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
        fn GetLastError() -> u32;
    }

    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

    pub fn detect() -> Option<Topology> {
        let mut needed: u32 = 0;
        unsafe {
            // First call sizes the buffer.
            let ok =
                GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, std::ptr::null_mut(), &mut needed);
            if ok == 0 && GetLastError() != ERROR_INSUFFICIENT_BUFFER {
                return None;
            }
            if needed == 0 {
                return None;
            }
            let mut buf: Vec<u8> = vec![0u8; needed as usize];
            let ok2 = GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                buf.as_mut_ptr(),
                &mut needed,
            );
            if ok2 == 0 {
                return None;
            }

            // Walk variable-length records.
            let mut cores: Vec<(u8, Vec<usize>)> = Vec::new();
            let mut offset = 0usize;
            while offset + mem::size_of::<u32>() * 2 <= needed as usize {
                let header = &*(buf.as_ptr().add(offset) as *const LogicalProcInfoEx);
                if header.relationship != RELATION_PROCESSOR_CORE {
                    offset += header.size as usize;
                    continue;
                }
                let efficiency = header.processor.efficiency_class;
                let group_count = header.processor.group_count as usize;
                let groups_ptr =
                    &header.processor.groups as *const GroupAffinity;
                let mut logical_ids = Vec::new();
                for g in 0..group_count {
                    let ga = &*groups_ptr.add(g);
                    let base = (ga.group as usize) * 64;
                    let mut mask = ga.mask;
                    while mask != 0 {
                        let bit = mask.trailing_zeros() as usize;
                        logical_ids.push(base + bit);
                        mask &= mask - 1;
                    }
                }
                cores.push((efficiency, logical_ids));
                offset += header.size as usize;
            }

            if cores.is_empty() {
                return None;
            }

            let max_class = cores.iter().map(|(c, _)| *c).max().unwrap_or(0);
            let min_class = cores.iter().map(|(c, _)| *c).min().unwrap_or(0);
            let is_hybrid = max_class != min_class;

            let mut perf: Vec<usize> = Vec::new();
            let mut eff: Vec<usize> = Vec::new();
            for (cls, lids) in &cores {
                if let Some(&first) = lids.first() {
                    // One thread per physical core (first SMT sibling) to avoid
                    // same-core contention.
                    if *cls == max_class {
                        perf.push(first);
                    } else if is_hybrid && *cls == min_class {
                        eff.push(first);
                    }
                }
            }
            perf.sort_unstable();
            perf.dedup();
            eff.sort_unstable();
            eff.dedup();
            Some(Topology {
                performance_cores: perf,
                efficiency_cores: eff,
                is_hybrid,
            })
        }
    }
}

// ----- Linux
#[cfg(target_os = "linux")]
mod linux {
    use super::Topology;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    fn read_u64(path: &Path) -> Option<u64> {
        fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
    }

    pub fn detect() -> Option<Topology> {
        let base = Path::new("/sys/devices/system/cpu");
        if !base.exists() {
            return None;
        }
        let mut entries: Vec<(usize, u64, usize)> = Vec::new(); // (cpu_id, weight, core_id)
        for ent in fs::read_dir(base).ok()? {
            let ent = match ent {
                Ok(e) => e,
                Err(_) => continue,
            };
            let name = ent.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("cpu") {
                continue;
            }
            let id_str = &name[3..];
            let cpu_id = match id_str.parse::<usize>() {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Prefer cpu_capacity (kernel ≥ 5.7 exposes this for hybrid SoCs); fall back to
            // cpuinfo_max_freq for older kernels and Intel.
            let cap_path = ent.path().join("cpu_capacity");
            let freq_path = ent.path().join("cpufreq/cpuinfo_max_freq");
            let weight = read_u64(&cap_path).or_else(|| read_u64(&freq_path))?;
            // Physical core id (so we can drop SMT siblings later).
            let core_path = ent.path().join("topology/core_id");
            let core_id = read_u64(&core_path).map(|v| v as usize).unwrap_or(cpu_id);

            entries.push((cpu_id, weight, core_id));
        }
        if entries.is_empty() {
            return None;
        }

        let max = entries.iter().map(|(_, w, _)| *w).max().unwrap_or(0);
        let min = entries.iter().map(|(_, w, _)| *w).min().unwrap_or(0);
        let is_hybrid = max != min;

        // Per-tier first-SMT-sibling per physical core.
        let mut perf_by_core: BTreeMap<usize, usize> = BTreeMap::new();
        let mut eff_by_core: BTreeMap<usize, usize> = BTreeMap::new();
        for &(cpu_id, w, core_id) in &entries {
            if w == max {
                perf_by_core.entry(core_id).or_insert(cpu_id);
            } else if is_hybrid && w == min {
                eff_by_core.entry(core_id).or_insert(cpu_id);
            }
        }
        let mut perf: Vec<usize> = perf_by_core.into_values().collect();
        perf.sort_unstable();
        let mut eff: Vec<usize> = eff_by_core.into_values().collect();
        eff.sort_unstable();
        Some(Topology {
            performance_cores: perf,
            efficiency_cores: eff,
            is_hybrid,
        })
    }
}
