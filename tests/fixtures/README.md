# Legacy serialization fixtures

These synthetic fixtures were generated with revision `444aba3` (version 0.8.1)
before changing the canonical short-key hash or standalone PGM serialization.
The generator used Rust 1.100.0 and the public builders and serialization APIs.
The tag-0 and tag-2 files retain the original serialized tables, wrapped in the
corresponding historical outer layouts.

Index and Hybrid input keys, in order:

```text
a
bc
hello
12345678
long-serialized-key
```

The static Index maps these keys to slots `[3, 0, 2, 4, 1]`. Hybrid returns the
original positions `[0, 1, 2, 3, 4]`. `index_v081.bin` and its mmap form use fp16
without Bloom; `index_tag0.bin` includes the historical mandatory Bloom filter.
The Hybrid fixture uses lean mode with a linear segment storing full hashes.

The raw-v4 PGM fixture contains sorted keys `[0, 2, 4, 1000, 1001]`.
Compatibility tests verify known positions and lookup behavior after loading,
ordinary reserialization and, for Index, aligned mmap reserialization.

`hot_tier_v081.bin` and `dynamic_v081.bin` were generated from the immutable
`b78b528` source, before exact hot-tier storage and maintenance policies changed.
The hot tier maps `[7, 11, 42, 99]` to `[100, 200, 300, 400]` with seed 17.
Its old payload contains no original-key table and remains probabilistic.
The dynamic fixture inserts `alice`, `bob`, `carol` with IDs 0, 1, 2, deletes
`bob`, and flushes. It retains `alice → 0`, `carol → 2`; the next new ID is 3.
Its configuration is flush threshold 2, maximum 4 tiers, fingerprints enabled
and sequential tier building.
