# Res-IPNM1-FlowNet Final Version

## Scope

The final selected model is now:

**Res-IPNM1-FlowNet = unified-rock ResNet encoder-decoder velocity predictor +
IPNM1 integrated-flux physics constraint.**

The model does not classify rock samples and does not use `rock_type`, rock
channels, or FiLM conditioning. It learns one geometry-to-flow mapping across
all rock samples.

The current HDF5 dataset contains single cross-section samples only, so this
work should be claimed as IPNM1. IPNM2 requires parent throat/sub-throat
grouping metadata that is not present in the current dataset.

## Method

Pipeline:

```text
cross-section geometry -> velocity field u(y,z) -> integrated q -> conductance g
```

Backbone:

```text
ResNet encoder-decoder
```

Loss:

```text
L = L_velocity_field + 0.10 * L_integrated_flux
```

The integrated-flux term makes the predicted velocity field consistent with the
IPNM1 flow rate and conductance, while still preserving the full local velocity
distribution for later transport or mass-transfer analysis.

## Final 3-Seed Experiment

Run directory:

```text
runs/res_ipnm1_flownet_100e_3seed_20260527
```

Settings:

- model: `res_ed`
- seeds: 42, 43, 44
- epochs: 100
- split: `runs/final_ipnm1_flownet_flux010_30e/split_info.json`
- loss: `field + 0.10 * integrated_flux`
- conductance postprocessing: `rho=1, ax=1e-4, segment_length=4,
  mu_lbm=0.5, target_mu=1`

Results:

| Metric | Mean +/- std |
| --- | ---: |
| Best epoch | 73.3 +/- 18.1 |
| Velocity pixel R2, all | 0.9875 +/- 0.0006 |
| Velocity pixel R2, validation | 0.9853 +/- 0.0007 |
| Velocity pixel R2, test | 0.9853 +/- 0.0004 |
| q mean relative error, all | 1.71% +/- 0.14% |
| q R2, all | 0.9831 +/- 0.0017 |
| Table conductance g mean relative error | 2.07% +/- 0.02% |
| Table conductance g R2 | 0.9980 +/- 0.0001 |
| True-velocity-derived g mean relative error | 1.71% +/- 0.14% |
| True-velocity-derived g R2 | 0.9986 +/- 0.0002 |

Per-seed details and figures:

- `runs/res_ipnm1_flownet_100e_3seed_20260527/res_ipnm1_3seed_metrics.csv`
- `runs/res_ipnm1_flownet_100e_3seed_20260527/res_ipnm1_3seed_loss_curves.png`
- `runs/res_ipnm1_flownet_100e_3seed_20260527/res_ipnm1_3seed_metric_summary.png`

## Architecture Comparison

A 93-epoch architecture comparison was run under:

```text
runs/arch_compare_93e_20260527
```

Summary:

| Model | Pixel R2 all | Pixel R2 val | q mean RE all | q R2 all | Table g mean RE | Table g R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet encoder-decoder | 0.9887 | 0.9857 | 1.38% | 0.9868 | 1.90% | 0.9982 |
| FNO | 0.9878 | 0.9866 | 1.90% | 0.9810 | 2.26% | 0.9977 |
| ConvNeXt-style encoder-decoder | 0.9877 | 0.9860 | 1.52% | 0.9868 | 2.06% | 0.9980 |
| Multi-task CNN | 0.9869 | 0.9861 | 1.95% | 0.9814 | 2.23% | 0.9977 |
| U-Net | 0.9860 | 0.9847 | 2.07% | 0.9778 | 2.31% | 0.9981 |
| IPNM1-FlowNet four-branch | 0.9858 | 0.9802 | 1.34% | 0.9800 | 2.29% | 0.9970 |
| DeepONet-lite | 0.6295 | 0.6296 | 7.21% | 0.7421 | 7.87% | 0.9736 |

Interpretation:

1. ResNet encoder-decoder provides the best overall balance across velocity
   field accuracy, q accuracy, and conductance accuracy.
2. The older four-branch IPNM1-FlowNet keeps a very interpretable
   velocity-to-conductance story, but the ResNet backbone is stronger as the
   final model.
3. FNO and ConvNeXt-style encoder-decoders remain useful ablations.
4. DeepONet-lite is not competitive in the current simple setup.

## Final 100-Epoch 3-Seed Comparison

All comparison models were also run for 100 epochs with seeds 42, 43, and 44.
The full combined archive is under:

```text
runs/final_100e_3seed_comparison_20260527
```

For the main manuscript comparison, keep only the selected rank 2-6 models:

1. Res-IPNM1-FlowNet
2. IPNM1-FlowNet four-branch
3. Multi-task CNN
4. ConvNeXt-style encoder-decoder
5. U-Net

Selected-model table:

```text
runs/final_100e_3seed_comparison_20260527/selected_rank2_6_summary.csv
```

| Rank | Model | Velocity R2 | q mean RE | q R2 | Table g mean RE | Table g R2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Res-IPNM1-FlowNet | 0.9875 +/- 0.0006 | 1.71% +/- 0.14% | 0.9831 +/- 0.0017 | 2.07% +/- 0.02% | 0.9980 +/- 0.0001 |
| 2 | IPNM1-FlowNet four-branch | 0.9862 +/- 0.0012 | 1.33% +/- 0.11% | 0.9811 +/- 0.0012 | 2.18% +/- 0.27% | 0.9972 +/- 0.0004 |
| 3 | Multi-task CNN | 0.9870 +/- 0.0001 | 1.87% +/- 0.11% | 0.9819 +/- 0.0016 | 2.18% +/- 0.12% | 0.9979 +/- 0.0002 |
| 4 | ConvNeXt-style encoder-decoder | 0.9867 +/- 0.0007 | 2.02% +/- 0.42% | 0.9788 +/- 0.0066 | 2.21% +/- 0.37% | 0.9979 +/- 0.0006 |
| 5 | U-Net | 0.9834 +/- 0.0039 | 3.02% +/- 1.44% | 0.9451 +/- 0.0528 | 3.29% +/- 1.64% | 0.9963 +/- 0.0029 |

Interpretation:

1. Res-IPNM1-FlowNet is the selected manuscript model because it has the best
   table-conductance error/R2 among the selected models and keeps a simple
   ResNet encoder-decoder plus IPNM1 physics story.
2. The older four-branch IPNM1-FlowNet keeps the lowest q mean relative error,
   but its velocity R2 and table g R2 are slightly weaker.
3. U-Net is clearly not the best final backbone after longer multi-seed
   comparison.
4. FNO and DeepONet-lite are kept in the full archive but excluded from the
   selected-model main figures.

Figures:

- `runs/final_100e_3seed_comparison_20260527/selected_rank2_6_r2_summary.png`
- `runs/final_100e_3seed_comparison_20260527/selected_rank2_6_error_summary.png`

Plotting TODO:

```text
TODO_PLOTTING.md
```

## Manuscript Claim

Recommended name:

**Res-IPNM1-FlowNet: a physics-guided velocity-to-conductance surrogate for
improved pore network modeling.**

Core innovation:

1. One unified model is used for all rock samples instead of rock classification.
2. The model predicts local velocity fields rather than directly fitting scalar
   conductance.
3. IPNM1 conductance is obtained by physically integrating the predicted
   velocity field into `q` and converting `q` to `g`.
4. Compared with scalar conductance fitting, the method keeps velocity
   distribution information for future transport, mass-transfer, or reactive
   flow extensions.
