# Draft: Methods and Results

## Methods

### Problem setting and overview

We developed Res-IPNM1-FlowNet as a physics-guided surrogate for single-phase flow prediction in improved pore network modelling. The aim is to predict the local velocity distribution in a throat cross-section and then recover the IPNM1 flow rate and hydraulic conductance through physical integration, rather than directly regressing a scalar conductance. This design preserves the velocity-field information required for subsequent transport or mass-transfer analysis while retaining the computational efficiency of a neural surrogate.

The workflow contains four steps. First, each throat cross-section is represented as a geometry tensor containing the pore mask and geometry descriptors. Second, a unified neural network predicts the local axial velocity field from this geometry without classifying rock type. Third, the predicted velocity field is integrated over the pore region to obtain the local flow rate. Finally, the flow rate is converted into IPNM1 conductance using the same post-processing convention as the reference data. This gives the mapping

```text
cross-section geometry -> velocity field -> integrated flow rate q -> conductance g.
```

The present work is restricted to IPNM1 because the current training dataset contains single cross-section samples and does not provide the parent-throat/sub-throat grouping required for IPNM2 network-scale aggregation. IPNM2 extension is therefore treated as a subsequent step.

### Geometry representation

For each sample, the input geometry is built from the pore cross-section. The primary channel is the binary pore mask, where pore pixels define the region available for flow and solid pixels are excluded from loss and integration. Additional channels encode geometric information derived from the cross-section, including distance-transform information, shape descriptors, local porosity-like features and coordinate-like descriptors. These channels provide the network with both local boundary information and global cross-sectional morphology.

The model is trained on all available rock samples together. No rock-type classifier, rock-specific branch or FiLM-style rock conditioning is used in the final model. This unified setting is intended to test whether a single geometry-to-flow operator can capture the dominant IPNM1 cross-section dependence without manually separating rock categories.

### Res-IPNM1-FlowNet architecture

The final network uses a ResNet-style encoder-decoder backbone. Given the geometry tensor, the encoder progressively extracts multi-scale geometric features, while the decoder reconstructs a dense velocity field at the original cross-section resolution. Residual blocks are used to stabilize feature propagation and improve training robustness. The output is a one-channel axial velocity field, which is masked by the pore region before physical integration.

This architecture was selected because the task requires dense spatial prediction rather than only scalar regression. Compared with a direct scalar model, the encoder-decoder output keeps the spatial velocity distribution. Compared with the earlier four-branch IPNM1-FlowNet, the residual encoder-decoder provides a simpler and more stable unified-rock backbone while preserving the same velocity-to-conductance physical pathway.

### Physics-guided IPNM1 loss

The training objective combines a pixel-level velocity-field loss with an integrated-flow constraint. The field term penalizes the difference between the predicted velocity field and the reference LBM velocity field over the pore pixels. The flux term penalizes the relative difference between the integrated predicted flow rate and the integrated reference flow rate. The loss is

```text
L = L_velocity_field + lambda_flux L_integrated_flux.
```

In the final model, `lambda_flux = 0.10`. This value was selected from an ablation study because it reduced q and conductance errors while maintaining high velocity-field accuracy. The flux term is used as an IPNM1 physical consistency constraint: it does not replace the velocity-field target, but biases the predicted field toward the correct integrated flow response.

### Flow-rate and conductance recovery

After inference, the predicted velocity field is converted to scalar hydraulic quantities by post-processing. The local flow rate is obtained by summing the predicted velocity over the pore region, with the same normalization recovery and area scaling used for the reference dataset. The conductance is then computed from the recovered flow rate using the IPNM1 post-processing convention applied to the teacher tables. We evaluate conductance against two references: conductance obtained by integrating the true velocity field, and conductance reported in the teacher table.

### Data and evaluation protocol

The main velocity-supervised experiments used `dataset_all_32.h5`, a preprocessed HDF5 training dataset rather than a trained model file. It was generated from the original simulation outputs and contains 1,728 cross-section samples with geometry tensors, normalized LBM velocity fields and sample metadata. Trained model weights are stored separately as `best.pt`, and inference outputs are stored as `predictions.npz`. The same train/validation/test split was used for all model comparisons and ablations. Model performance was evaluated using pore-pixel velocity R2, integrated flow-rate relative error and R2, and conductance relative error and R2.

We also audited the teacher-provided `300x200x200_data` IPNM1 dataset to verify the original cross-section and conductance resources. The valid cross-section counts were 1,255 for Bead, 1,719 for Benth, 1,560 for Berea_ICL and 1,267 for Font18. All valid cross-sections matched conductance entries by local identifier. Some conductance files also contain rows associated with disabled or invalid cross-sections, which were not treated as valid velocity-supervised samples.

## Results

### Res-IPNM1-FlowNet accurately predicts velocity fields and IPNM1 conductance

Across three random seeds and 100 training epochs, Res-IPNM1-FlowNet achieved high velocity-field accuracy and stable conductance recovery. The pore-pixel velocity R2 was `0.9875 +/- 0.0006`, showing that the model learned the local velocity distribution rather than only the integrated flow response. The integrated flow-rate mean relative error was `1.71% +/- 0.14%`, with q R2 of `0.9831 +/- 0.0017`.

The recovered conductance was also consistent with the teacher table. The table-conductance mean relative error was `2.07% +/- 0.02%`, and the table-conductance R2 was `0.9980 +/- 0.0001`. These results show that the velocity-first route can recover scalar IPNM1 conductance with high accuracy while preserving local velocity information.

### The integrated-flux constraint improves conductance consistency

We tested the effect of the IPNM1 flux-consistency term by varying `lambda_flux` while keeping the same ResNet encoder-decoder backbone. Without the flux term (`lambda_flux = 0`), the model maintained high velocity R2 (`0.9879`) but produced a larger table-conductance error (`3.01%`). A small flux weight (`lambda_flux = 0.01`) reduced the conductance error to `2.50%`. The best setting was `lambda_flux = 0.10`, which achieved `1.67%` q relative error and `2.05%` table-conductance relative error, with velocity R2 remaining high at `0.9880`.

Increasing the flux weight to `1.00` did not further improve conductance accuracy and reduced velocity-field R2 to `0.9795`. This indicates that the flux term is useful when used as a moderate physical regularizer, but an overly strong scalar constraint can compromise local velocity-field fidelity.

### Direct scalar fitting does not provide the same evidence as velocity-field prediction

To test whether the task could be replaced by scalar regression, we trained direct scalar baselines to predict either the integrated reference flow rate or the table-derived flow/conductance label. Direct fitting to `q_true` reached an all-sample mean relative error of `2.22% +/- 0.13%` and test error of `2.68% +/- 0.02%`. This confirms that the integrated scalar is learnable from geometry. However, this model does not output a velocity field and therefore cannot support downstream analyses that depend on spatial velocity distribution.

Direct fitting to the table-derived scalar was much less stable. The all-sample mean relative error was `535.55% +/- 191.42%`, and the test mean relative error was `1325.52% +/- 511.75%`. This poor stability supports the use of the velocity-field-first IPNM1 route: the model learns a physically interpretable intermediate field and then derives q and g through an explicit integration step, rather than directly absorbing all table-level variability into a scalar regressor.

### Teacher IPNM1 data support the cross-section-based modelling setup

The original `300x200x200_data` resource contains rock-specific IPNM1 cross-section files and PNM conductance tables. The audit confirmed that valid cross-sections can be matched to conductance labels by local identifier. The median pore-pixel counts differed across rock types, from 29 in Berea_ICL and 33 in Benth to 126 in Bead, indicating substantial geometric variability across the teacher data. This supports the need for a geometry-driven model rather than a rock-class-specific lookup strategy.

The current experiments use IPNM1-level cross-section supervision and conductance recovery. Because the teacher data also contain PNM network files, they provide a natural route for future IPNM2 extension. That extension will require explicit sub-throat grouping, series aggregation of sub-throat conductances and network-scale pressure/flow solving, which are outside the present IPNM1 evaluation.

### Summary of experimental evidence

Together, the experiments support three claims. First, Res-IPNM1-FlowNet predicts LBM-like local velocity fields with high pore-pixel R2. Second, the predicted velocity field can be physically integrated into accurate IPNM1 q and conductance values. Third, a moderate integrated-flux constraint improves q/g consistency without sacrificing local velocity accuracy. These results justify the proposed velocity-to-conductance surrogate as an IPNM1 improvement over direct scalar fitting, particularly when velocity distribution is needed for later transport or mass-transfer modelling.

## Suggested Figures and Tables

- Figure 1: Method schematic, geometry -> velocity field -> q -> g.
- Figure 2: Representative true/predicted/error velocity maps.
- Figure 3: q and g parity plots from `conductance_results.csv`.
- Figure 4: Flux-loss ablation using `flux_ablation_summary.csv`.
- Table 1: Main three-seed Res-IPNM1-FlowNet metrics.
- Table 2: Scalar baseline and flux ablation results.

## Notes

- Replace `+/-` with journal-preferred notation during final formatting.
- Add exact dataset split counts if needed in the final Methods.
- Add inference-time comparison once timing experiments are available.
