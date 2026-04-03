# Alpha-Chain Seed Summary

Current reference setup:

- diagnostics source: `diagnostics/alpha_chain_pilot/adamw/alpha_chain_diagnostics`
- this summary reflects the results currently present on disk; at the moment that includes `seed_13` and `seed_17`
- `Full grad projection` reports the range across `w_seed`
- for all lag-specific columns, each `w_seed` is first averaged across the four lags, then the table reports the range across those lag-averaged `w_seed` values
- each cell reports `ECF range / McCulloch range`
- all estimates in the current `seed_13` and `seed_17` runs are marked reliable

| seed | model | # w_seeds | Full grad projection (ECF / MCC) | Exact lag projection, inst. (ECF / MCC) | Exact lag projection, seq. avg. (ECF / MCC) | Matched statistic, inst. (ECF / MCC) | Matched statistic, seq. avg. (ECF / MCC) | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | diag | 4 | [1.9952, 2.0000] / [1.9627, 2.0000] | [1.2969, 1.4423] / [1.0838, 1.1882] | [1.9043, 1.9347] / [1.8614, 1.9185] | [1.3242, 1.7219] / [1.1333, 1.4591] | [1.8643, 1.9514] / [1.7232, 1.8794] | Within this seed, `w_seed` variability is modest for the sequence-level objects. Temporal averaging strongly regularizes both ladders, and the final sequence-level statistics remain fairly close to the projected full-gradient range. |
| 13 | gru | 4 | [1.9923, 1.9989] / [1.9313, 2.0000] | [1.3490, 1.3908] / [1.1200, 1.1327] | [1.7336, 1.8021] / [1.6074, 1.7066] | [1.4021, 1.4723] / [1.1723, 1.2037] | [1.9366, 1.9554] / [1.8754, 1.8841] | The main GRU split in this seed is not projection instability: the exact lag-projected sequence average stays materially heavier-tailed than the full-gradient projection, while the first-order matched sequence average shifts back close to the full-gradient range. |
| 17 | diag | 8 | [1.6202, 1.7727] / [1.6302, 1.7697] | [1.1236, 1.2602] / [1.0308, 1.0943] | [1.3573, 1.4835] / [1.3093, 1.4412] | [1.5272, 1.7856] / [1.2368, 1.5225] | [1.7867, 1.9509] / [1.7193, 1.9230] | `w_seed` variability is again modest relative to the much larger shift across training seed. Temporal averaging remains a strong regularizer, and the matched sequence average is clearly more regular than the exact lag-projected sequence average. |
| 17 | gru | 8 | [1.9929, 2.0000] / [1.9567, 2.0000] | [1.3342, 1.4375] / [1.0979, 1.1755] | [1.9239, 1.9634] / [1.8761, 1.9053] | [1.3306, 1.5656] / [1.0896, 1.2742] | [1.9127, 1.9765] / [1.8689, 1.9545] | In this seed, both sequence-level objects are close to the projected full-gradient regime. The matched sequence average remains at least as regular as the exact sequence average, and `w_seed` variability stays small. |

## Reading Notes

- **What is confirmed across seeds:** the broad five-object mechanism survives. In both architectures and both seeds, the instantaneous lag-specific objects are clearly heavier-tailed than their sequence-averaged counterparts, so temporal averaging is a robust regularizer.
- **What is also confirmed:** the first-order matched sequence average has tail behavior broadly comparable to the exact lag-projected sequence average. Across both seeds, it is often slightly more regular, but the main point is qualitative consistency rather than a one-sided bound.
- **What is not confirmed as seed-invariant:** the absolute alpha levels and the detailed ordering between objects `1`, `3`, and `5`. The biggest shift is `diag`, where the projected full-gradient probe moves from a near-Gaussian range in `seed_13` to a clearly heavier-tailed range in `seed_17`.
- **What this means for the paper:** the appendix can credibly argue that the one-dimensional matched statistic is a usable and stable proxy for the relevant lag-specific transport object, but it should not claim a seed-invariant quantitative propagation law from the projected full gradient to the final matched statistic.
- **Projection variability vs seed variability:** within each seed, the `w_seed` ranges are modest for the sequence-level objects. The larger source of variation is the training seed, not the choice among the tested one-dimensional probes.
