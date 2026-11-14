use kira_kv_engine::{
    HybridBuilder, Index, IndexBuilder, IndexConfig, IndexError, MphConfig, PgmBuilder, PgmError,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct ThreadObservations {
    caller: std::thread::ThreadId,
    foreign_thread: AtomicBool,
    reads: AtomicUsize,
    drops: AtomicUsize,
}

impl ThreadObservations {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            caller: std::thread::current().id(),
            foreign_thread: AtomicBool::new(false),
            reads: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
        })
    }

    fn observe_thread(&self) {
        if std::thread::current().id() != self.caller {
            self.foreign_thread.store(true, Ordering::Relaxed);
        }
    }
}

struct ObservedKey {
    bytes: [u8; 8],
    observations: Arc<ThreadObservations>,
}

impl AsRef<[u8]> for ObservedKey {
    fn as_ref(&self) -> &[u8] {
        self.observations.observe_thread();
        self.observations.reads.fetch_add(1, Ordering::Relaxed);
        &self.bytes
    }
}

impl Drop for ObservedKey {
    fn drop(&mut self) {
        self.observations.observe_thread();
        self.observations.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn sequential_owned_build_hashes_and_drops_on_caller_thread() {
    #[cfg(feature = "parallel")]
    {
        // The other tests may legitimately initialize Rayon. A fresh process lets
        // this public-path regression detect every accidental global-pool phase
        // without changing process-global environment variables in the test suite.
        const CHILD_FLAG: &str = "KIRA_API_TEST_SEQUENTIAL_CHILD";
        if std::env::var_os(CHILD_FLAG).is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "sequential_owned_build_hashes_and_drops_on_caller_thread",
                    "--nocapture",
                ])
                .env(CHILD_FLAG, "1")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "child test failed:\n{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
    }
    // Cross the Bloom, partition-scatter, multi-part MPH and parallel-drop
    // thresholds. This catches accidental use of Rayon's global pool.
    let count = (1 << 18) + 1;
    let observations = ThreadObservations::new();
    let keys = (0..count)
        .map(|value| ObservedKey {
            bytes: (value as u64).to_le_bytes(),
            observations: Arc::clone(&observations),
        })
        .collect();
    let index = IndexBuilder::new().with_parallel_build(false).build_index(keys).unwrap();
    assert_eq!(index.len(), count);
    assert!(observations.reads.load(Ordering::Relaxed) >= count);
    assert_eq!(observations.drops.load(Ordering::Relaxed), count);
    assert!(!observations.foreign_thread.load(Ordering::Relaxed));
    for value in [0, 1, count / 2, count - 1] {
        assert!(index.lookup(&(value as u64).to_le_bytes()).unwrap() < count);
    }
    let numeric_keys: Vec<_> = (0..count as u64).collect();
    let pgm = PgmBuilder::new()
        .with_parallel(false)
        .with_bloom_filter(true)
        .build(numeric_keys.clone())
        .unwrap();
    let hybrid = HybridBuilder::new().with_parallel(false).build_from_u64(&numeric_keys).unwrap();
    for value in [0, 1, count / 2, count - 1] {
        assert_eq!(pgm.index(value as u64).unwrap(), value);
        assert_eq!(hybrid.lookup_u64(value as u64).unwrap(), value as u32);
    }
    #[cfg(feature = "parallel")]
    assert!(
        rayon::ThreadPoolBuilder::new().num_threads(1).build_global().is_ok(),
        "sequential build initialized Rayon's global pool"
    );
}

#[test]
fn invalid_geometry_returns_typed_errors_before_reading_keys() {
    let observations = ThreadObservations::new();
    let keys = [ObservedKey { bytes: 7u64.to_le_bytes(), observations: Arc::clone(&observations) }];
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let config = IndexConfig {
            mph_config: MphConfig { lambda: value, ..MphConfig::default() },
            ..IndexConfig::default()
        };
        assert!(matches!(Index::build_index_ref(&keys, config), Err(IndexError::InvalidConfig(_))));
    }
    for value in [0.0, -1.0, 1.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let config = IndexConfig {
            mph_config: MphConfig { alpha: value, ..MphConfig::default() },
            ..IndexConfig::default()
        };
        assert!(matches!(Index::build_index_ref(&keys, config), Err(IndexError::InvalidConfig(_))));
    }
    for config in [
        MphConfig { lambda: f64::MIN_POSITIVE, ..MphConfig::default() },
        MphConfig { alpha: f64::MIN_POSITIVE, ..MphConfig::default() },
    ] {
        assert!(matches!(
            IndexBuilder::new().with_mph_config(config).build_index_ref(&keys),
            Err(IndexError::CapacityExceeded(_))
        ));
    }
    assert_eq!(observations.reads.load(Ordering::Relaxed), 0);
}

#[test]
fn empty_build_validates_configuration_and_accepts_valid_config() {
    let keys: &[[u8; 8]] = &[];
    assert!(IndexBuilder::new().build_index_ref(keys).unwrap().is_empty());
    let config = MphConfig { alpha: f64::NAN, ..MphConfig::default() };
    assert!(matches!(
        IndexBuilder::new().with_mph_config(config).build_index_ref(keys),
        Err(IndexError::InvalidConfig(_))
    ));
}

#[test]
fn named_public_configuration_and_pgm_error_types_are_usable() {
    let config = MphConfig { lambda: 2.5, alpha: 0.9, ..MphConfig::default() };
    let keys: Vec<_> = (0..100u64).map(u64::to_le_bytes).collect();
    let index = IndexBuilder::new()
        .with_mph_config(config)
        .with_parallel_build(false)
        .build_index_ref(&keys)
        .unwrap();
    let mut slots: Vec<_> = keys.iter().map(|key| index.lookup(key).unwrap()).collect();
    slots.sort_unstable();
    assert_eq!(slots, (0..keys.len()).collect::<Vec<_>>());
    let result: Result<_, PgmError> = PgmBuilder::new().build(vec![1, 1]);
    assert!(matches!(result, Err(PgmError::DuplicateKeys)));
}

#[test]
fn low_level_result_builder_rejects_invalid_geometry_without_panicking() {
    use kira_kv_engine::__internal::{Builder, PtrHash25Error};
    assert!(matches!(Builder::new().build(&[]), Err(PtrHash25Error::Unresolvable)));
    for config in [
        MphConfig { lambda: f64::NAN, ..MphConfig::default() },
        MphConfig { alpha: 0.0, ..MphConfig::default() },
        MphConfig { lambda: f64::MIN_POSITIVE, ..MphConfig::default() },
        MphConfig { alpha: f64::MIN_POSITIVE, ..MphConfig::default() },
    ] {
        assert!(matches!(
            Builder::new().with_config(config).build(&[7]),
            Err(PtrHash25Error::Unresolvable)
        ));
    }
}
