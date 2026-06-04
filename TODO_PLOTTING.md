# IPNM1 plotting data map

This file records the current paper-facing data sources after the IPNM1
experiment cleanup.

## Main Method

- Paper name: Ours
- Checkpoints:
  `runs/lambda_selected_3seed_20260529/flux_0p2__dist_0p005__seed_<seed>/res_ed/best.pt`
- Final loss:
  `L = L_field + 0.20 L_flux + 0.005 L_dist`
- Seeds: 42, 43, 44

## Figure 1: Method Overview

Show:

- local throat cross-section geometry
- velocity-field prediction
- physical integration to local flow rate q
- conductance conversion
- conductance backfill into teacher PNM pressure solver for global K validation

Non-data figure description:

- `makefig/fig1_method_overview.txt`
- `makefig/fig2_res_ipnm1_flownet_architecture.txt`
- `makefig/fig3_physics_guided_training_strategy.txt`
- `makefig/fig4_ipnm1_to_ipnm2_extension.txt`

## Figure 2: Dataset and Preprocessing

Use:

- Dataset split table:
  `runs/paper_six_model_full_comparison_20260530/dataset_split_summary.csv`
- Teacher IPNM1 audit:
  `runs/ipnm1_manuscript_experiments_20260527/teacher_ipnm1_audit_summary.csv`
- Training HDF5:
  `dataset_all_32.h5`
- Teacher geometry conversion code:
  `code/convert_teacher_cross_sections.py`

Recommended panels:

- samples per rock/source
- pore-area or pore-fraction distribution
- one or two example normalized cross-sections

## Figure 3: Main Six-Model Comparison

Use:

- Overall six-model table:
  `runs/paper_six_model_full_comparison_20260530/six_model_full_comparison.csv`
- Paper-ready README:
  `runs/paper_six_model_full_comparison_20260530/README.md`

Models:

- Ours
- IPNM-FlowNet
- FNO
- ConvNeXt-ED
- Multi-task CNN
- U-Net

Metrics:

- global K mean relative error
- global K max relative error
- local q mean relative error
- local q R2
- velocity-field R2

Do not include throat coverage in paper figures.

## Figure 4: Ours Global K Stability

Use:

- Three-seed global rows:
  `runs/ours_global_3seed_20260530/ours_global_3seed_rows.csv`
- Three-seed global summary:
  `runs/ours_global_3seed_20260530/ours_global_3seed_summary.csv`
- Paper-ready README:
  `runs/ours_global_3seed_20260530/README.md`

Recommended panels:

- predicted K vs teacher K for Bead, Benth, Berea, Font
- per-rock K relative error with seed error bars
- note internally that Benth and Berea PNM files are identical in
  `300x200x200_data/PNM_simulation`; avoid overinterpreting their difference.

## Figure 5: Lambda Selection

Use:

- 2D lambda grid:
  `runs/lambda_grid_seed42_20260529/lambda_grid_summary.csv`
- Three-seed selected lambda confirmation:
  `runs/lambda_selected_3seed_20260529/selected_lambda_group_summary.csv`

Recommended panels:

- heatmap of table-g relative error over `lambda_flux` and `lambda_dist`
- grouped bar plot for selected candidates across three seeds

## Figure 6: Efficiency

Use:

- Raw timing:
  `runs/ours_global_3seed_20260530/ours_global_3seed_timing.csv`
- Aggregated timing:
  `runs/ours_global_3seed_20260530/ours_inference_efficiency_summary.csv`

Report:

- Ours inference averages `3.90 +/- 0.37 ms` per throat cross-section on the
  current GPU/PyTorch environment.
- If teacher LBM runtime is later available, add direct speedup ratio here.

## Supplementary Figures

Use these only as supporting/appendix evidence:

- ARS ablation:
  `runs/ars_ablation_seed42_100e_20260529/ars_ablation_summary.csv`
- Dynamic lambda ablation:
  `runs/adaptive_lambda_ablation_seed42_100e_20260529/adaptive_lambda_summary.csv`
- Dist/Poisson loss ablation:
  `runs/loss_structure_ablation_seed42_100e_20260527/loss_structure_ablation_summary.csv`
- Direct scalar baseline:
  `runs/ipnm1_manuscript_experiments_20260527/scalar_baseline_summary.csv`

These are not final-method claims unless explicitly discussed as ablation or
negative evidence.
