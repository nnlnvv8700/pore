# IPNM1 experiment completion note

This note records how the four experiment gaps were addressed.

## 1. Ours global permeability stability

Three-seed global permeability validation has been added for Ours by
backfilling predicted throat conductance into the teacher PNM pressure solver.

Data:

- `runs/ours_global_3seed_20260530/ours_global_3seed_rows.csv`
- `runs/ours_global_3seed_20260530/ours_global_3seed_summary.csv`
- `runs/ours_global_3seed_20260530/README.md`

Main result:

| Rock | K relative error |
| --- | ---: |
| Bead | 0.80% +/- 0.51% |
| Benth | 5.14% +/- 0.15% |
| Berea | 0.39% +/- 0.48% |
| Font | 4.38% +/- 0.53% |
| Mean | 2.68% +/- 2.23% |

Interpretation:

- Benth and Font remain the harder global PNM cases across seeds.
- Their errors are stable, not a seed-42 accident.
- The predicted K values are consistently higher than teacher K, indicating a
  small positive conductance bias amplified by low-permeability or bottlenecked
  networks.

## 2. Computation efficiency

Ours inference timing on teacher IPNM1 geometries has been measured during the
three-seed global run.

Data:

- `runs/ours_global_3seed_20260530/ours_global_3seed_timing.csv`
- `runs/ours_global_3seed_20260530/ours_inference_efficiency_summary.csv`

Main result:

- Average neural inference time: `3.90 +/- 0.37 ms` per throat cross-section.
- This number includes PyTorch dataloader/model inference overhead in the
  current `nnlnvv` GPU environment.
- A direct speedup ratio against teacher LBM should be added only if the
  original per-throat LBM runtime is measured or provided.

## 3. Dataset and split statistics

The dataset split table has been generated.

Data:

- `runs/paper_six_model_full_comparison_20260530/dataset_split_summary.csv`

Summary:

| Source | Total | Train | Val | Test |
| --- | ---: | ---: | ---: | ---: |
| Bead | 700 | 560 | 70 | 70 |
| Benth | 119 | 95 | 11 | 13 |
| Berea_ICL | 368 | 294 | 36 | 38 |
| Font18 | 268 | 214 | 26 | 28 |
| rock_type_4 | 221 | 176 | 22 | 23 |
| rock_type_5 | 52 | 41 | 5 | 6 |
| All | 1728 | 1380 | 170 | 178 |

The split keeps a single unified model and does not use rock classification,
rock-type channels, or FiLM conditioning.

## 4. Figure and paper-data map

The plotting TODO has been rewritten to match the current IPNM1 paper route.

Updated file:

- `TODO_PLOTTING.md`

Main figures should use:

- six-model comparison:
  `runs/paper_six_model_full_comparison_20260530/six_model_full_comparison.csv`
- Ours three-seed global K:
  `runs/ours_global_3seed_20260530/ours_global_3seed_summary.csv`
- lambda selection:
  `runs/lambda_grid_seed42_20260529/lambda_grid_summary.csv`
  and
  `runs/lambda_selected_3seed_20260529/selected_lambda_group_summary.csv`
- efficiency:
  `runs/ours_global_3seed_20260530/ours_inference_efficiency_summary.csv`

Throat coverage should not be shown as a paper metric.

## Important caveat

`300x200x200_data/PNM_simulation/Benth` and
`300x200x200_data/PNM_simulation/Berea` have identical PNM files for topology
and teacher conductance. Therefore, the global PNM comparison for these two
folders should be interpreted cautiously. Local geometry-level metrics remain
valid, but the global PNM-level comparison is constrained by the provided
teacher files.
