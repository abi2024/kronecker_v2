# Run ledger

Append one row per run. This file becomes the methods section of the paper.
A number without a manifest.json is not a result.

| date | milestone | config | seed | git SHA | codec fingerprint | final loss | outputs |
|------|-----------|--------|------|---------|-------------------|-----------|---------|
| 2026-08-14 | m2 | m2_onehot | 1337 | 7695669 | onehot16 | 4.8129 | results\m2\m2_onehot |
| 2026-08-14 | m2 | m2_wave_sqrtlen | 1337 | 7695669 | wave2048_sqr | 4.8612 | results\m2\m2_wave_sqrtlen |
| 2026-08-14 | m2 | m2_wave_l2 | 1337 | 7695669 | wave2048_l2 | 4.7532 | results\m2\m2_wave_l2 |
| 2026-08-14 | m2 | m2_wave_znorm | 1337 | 7695669 | wave2048_zno | 4.9023 | results\m2\m2_wave_znorm |
| 2026-08-14 | m2 | m2_dense | 1337 | 7695669 | dense | 4.5592 | results\m2\m2_dense |

## M4 dose-response — prediction registered before the runs
Matched (one-hot, wave/l2) pairs at pos_dim 12/16/24/32. Known dose from the M1
audit: 5,335 / 903 / 93 / 0 tokens permanently merged. pos_dim=16 is M2's pair.
PREDICTION: the wave-minus-one-hot gap shrinks monotonically as pos_dim rises,
and vanishes at pos_dim=32 where the tokenizer guarantees zero collisions.
pos_dim=32 is a placebo dose we did not construct — if the effect survives there,
it was never about collisions and M2's mechanism claim is wrong.
| 2026-08-14 | m2 | m4_p12_onehot | 1337 | a000c6d | onehot12 | 4.8729 | results\m4\m4_p12_onehot |
| 2026-08-14 | m2 | m4_p12_wave | 1337 | a000c6d | wave1536_l2 | 4.7405 | results\m4\m4_p12_wave |
| 2026-08-14 | m2 | m4_p24_onehot | 1337 | a000c6d | onehot24 | 4.8074 | results\m4\m4_p24_onehot |
| 2026-08-14 | m2 | m4_p24_wave | 1337 | a000c6d | wave3072_l2 | 4.7570 | results\m4\m4_p24_wave |
| 2026-08-14 | m2 | m4_p32_onehot | 1337 | a000c6d | onehot32 | 4.8214 | results\m4\m4_p32_onehot |
| 2026-08-15 | m2 | m4_p32_wave | 1337 | a000c6d | wave4096_l2 | 4.7600 | results\m4\m4_p32_wave |
| 2026-08-17 | m2 | m5_s1338_onehot | 1338 | a000c6d | onehot16 | 4.8122 | results\m5\m5_s1338_onehot |
| 2026-08-17 | m2 | m5_s1338_wave | 1338 | a000c6d | wave2048_l2 | 4.7343 | results\m5\m5_s1338_wave |
| 2026-08-17 | m2 | m5_s1339_onehot | 1339 | a000c6d | onehot16 | 4.8217 | results\m5\m5_s1339_onehot |
| 2026-08-17 | m2 | m5_s1339_wave | 1339 | a000c6d | wave2048_l2 | 4.7588 | results\m5\m5_s1339_wave |
| 2026-08-17 | m2 | m5_rp | 1337 | a000c6d | rp16_l2 | 4.7612 | results\m5\m5_rp |
| 2026-08-17 | m2 | m5_bag | 1337 | a000c6d | bag2048_l2 | 4.8400 | results\m5\m5_bag |
| 2026-08-17 | m2 | m5_wave1024 | 1337 | a000c6d | wave1024_l2 | 4.7552 | results\m5\m5_wave1024 |
| 2026-08-17 | m2 | m5_wave768 | 1337 | a000c6d | wave768_l2 | 4.7152 | results\m5\m5_wave768 |
