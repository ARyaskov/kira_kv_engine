use kira_kv_engine::__internal::{CompressedPilots, CompressedPilotsV2};

#[test]
fn round_trip_all_small() {
    let pilots: Vec<u8> = (0..10000).map(|i| (i % 15) as u8).collect();
    let c = CompressedPilots::from_flat(&pilots);
    assert_eq!(c.overflow_count(), 0);
    for (i, &p) in pilots.iter().enumerate() {
        assert_eq!(c.get(i), p, "mismatch at {i}");
    }
}

#[test]
fn round_trip_with_overflow() {
    let pilots: Vec<u8> = (0..10000)
        .map(|i| if i % 7 == 0 { (200 + (i % 50)) as u8 } else { (i % 12) as u8 })
        .collect();
    let c = CompressedPilots::from_flat(&pilots);
    assert!(c.overflow_count() > 1000);
    for (i, &p) in pilots.iter().enumerate() {
        assert_eq!(c.get(i), p, "mismatch at {i}: expected {p}");
    }
}

#[test]
fn v2_round_trip_skewed() {
    let n = 50_000;
    let mut pilots = vec![0u8; n];
    for i in 0..n {
        pilots[i] = if i % 4 == 0 {
            if i % 20 == 0 {
                (15 + (i % 240)) as u8
            } else {
                1 + ((i / 4) % 14) as u8
            }
        } else {
            0
        };
    }
    let c = CompressedPilotsV2::from_flat(&pilots);
    for (i, &p) in pilots.iter().enumerate() {
        assert_eq!(c.get(i), p, "mismatch at {i}");
    }
    let mem = c.memory_usage();
    assert!(mem < n / 2, "v2 memory {mem} not better than flat/2 ({})", n / 2);
}

#[test]
fn v2_all_zeros() {
    let pilots = vec![0u8; 1000];
    let c = CompressedPilotsV2::from_flat(&pilots);
    for i in 0..1000 {
        assert_eq!(c.get(i), 0);
    }
}

#[test]
fn v2_all_overflow() {
    let pilots: Vec<u8> = (0..1000).map(|i| 100 + (i % 156) as u8).collect();
    let c = CompressedPilotsV2::from_flat(&pilots);
    for (i, &p) in pilots.iter().enumerate() {
        assert_eq!(c.get(i), p);
    }
}

#[test]
fn memory_savings_at_scale() {
    let n = 100_000;
    let pilots: Vec<u8> = (0..n)
        .map(|i| {
            if i * 1000 % n < 50 {
                (16 + (i % 240)) as u8
            } else {
                (i % 15) as u8
            }
        })
        .collect();
    let c = CompressedPilots::from_flat(&pilots);
    let flat_size = pilots.len();
    let compressed = c.memory_usage();
    assert!(compressed < flat_size * 7 / 10, "compressed = {compressed}, flat = {flat_size}");
    for (i, &p) in pilots.iter().enumerate() {
        assert_eq!(c.get(i), p);
    }
}
