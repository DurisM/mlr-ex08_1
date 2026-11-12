# Task 4 — a better leaf classifier

**Agent-generated measurements draft, 12 November 2026.** The student must revise
this interpretation in their own words and add actual phone-camera observations
before submitting the final report.

The code keeps Task 3's `LeafDataset`, `LeafNetBN` and training loop. Both classes
are unchanged; no new network classes, optimizer types or regularization layers
were introduced. The fixed baseline is Task 3d: batch-normalized CNN with geometric
augmentation, Adam at 0.001, batch size 32 and 15 epochs. `cnn_plain` removes only
geometry and controls preprocessing before the MobileNet comparison. MobileNet's
new four-output final layer is either the only trainable layer or the whole
network is fine-tuned. The frozen case also freezes pretrained batch-normalization
statistics and disables pretrained dropout. Architecture/pretraining changes are
a model-family comparison; their individual effects cannot be separated here.

All 11 setups used the same 136 training and 37 validation photographs and seeds
0, 1 and 2: **33 retained training runs**. Class order is apple, cherry, chestnut,
maple. Only training images are augmented. Geometry is the existing 60–100% random
crop, horizontal/vertical flips and ±20° rotation. Colour adds
`ColorJitter(0.25, 0.25, 0.25, 0.05)`. Cosine scheduling uses `T_max` equal to the
15- or 30-epoch budget. All rows use final-epoch weights, not a selected best epoch.
The 30-epoch controls reproduced exactly the first 15 epochs of their short
controls for every seed.

Camera-like validation consists of the 37 validation images randomly cropped,
rotated and colour-jittered **once with seed 2026**, saved and reused for every
run. It measures robustness to those transformations, not independent new leaves
or actual phone-camera accuracy. Raw per-seed results, histories and per-image
predictions are in `task4_results`; `comparison_table.md` also gives every change
against the fixed baseline. The direct controls below isolate the relevant
individual change; the last two rows deliberately combine changes after the
initial 27 runs. Their choices are recorded in `combination_plan.json`.

Values are mean ± sample SD across three seeds. Deltas are paired by seed;
**pp** means percentage points. Control names match `task4.py` and `results.csv`.

| Model / change | Clean % ± SD | Camera-like % ± SD | Direct control | Paired Δ camera, pp ± SD |
| --- | ---: | ---: | --- | ---: |
| Task 3 BN + geometry (fixed baseline) | 66.67 ± 1.56 | 57.66 ± 4.13 | baseline | 0.00 ± 0.00 |
| Task 3 BN without geometry (control) | 83.78 ± 2.70 | 57.66 ± 4.13 | baseline | 0.00 ± 7.15 |
| Task 3 BN + geometry + colour | 69.37 ± 4.13 | 72.07 ± 1.56 | baseline | 14.41 ± 4.13 |
| MobileNet frozen, final layer only | 95.50 ± 1.56 | 85.59 ± 4.13 | cnn_plain | 27.93 ± 4.13 |
| MobileNet fully fine-tuned | 71.17 ± 5.63 | 62.16 ± 9.74 | mobile_frozen | -23.42 ± 13.60 |
| Fine-tuned + geometry | 81.98 ± 4.13 | 73.87 ± 8.26 | mobile_finetuned | 11.71 ± 15.37 |
| Fine-tuned + geometry + colour | 77.48 ± 11.25 | 78.38 ± 11.78 | mobile_geometry | 4.50 ± 5.63 |
| Fine-tuned + cosine, 15 epochs | 75.68 ± 9.74 | 72.97 ± 12.39 | mobile_finetuned | 10.81 ± 21.62 |
| Fine-tuned, 30 epochs | 81.98 ± 3.12 | 68.47 ± 1.56 | mobile_finetuned | 6.31 ± 11.25 |
| Fine-tuned + geometry + colour + cosine, 30 epochs | 88.29 ± 4.13 | 85.59 ± 3.12 | mobile_geometry_colour | 7.21 ± 14.89 |
| Frozen + geometry + colour + cosine, 30 epochs | 84.68 ± 3.12 | 83.78 ± 0.00 | mobile_frozen | -1.80 ± 4.13 |

Colour helped the Task 3 CNN's camera-like accuracy by **14.41 ± 4.13 pp**, with a
positive change in all three seeds. Its clean gain was only **2.70 ± 5.41 pp**,
which is not convincing. Removing geometry improved clean CNN accuracy but left
mean camera-like accuracy unchanged, showing why the clean score alone is not
enough for the camera goal.

Frozen MobileNet improved both metrics substantially. Full fine-tuning at the
tested learning rate did **not** help: camera-like accuracy fell by
**23.42 ± 13.60 pp**, in all three seeds. Geometric augmentation recovered
**11.71 ± 15.37 pp** for the fine-tuned model. Adding colour on top recovered
another **4.50 ± 5.63 pp**, but reduced clean accuracy. Cosine scheduling alone
added **10.81 ± 21.62 pp**; 30 epochs alone added **6.31 ± 11.25 pp**. Those last
gains varied substantially by seed and are not strong evidence on their own.

The full fine-tuning recipe reached 85.59% camera-like accuracy, tying frozen
MobileNet, but clean accuracy remained lower (88.29% versus 95.50%). Applying
the same recipe to the frozen network did not help: camera-like accuracy was
83.78%, and clean accuracy fell to 84.68%. The combinations therefore did not
beat the simpler frozen setup under the stated selection rule.

One validation image is **2.70 pp**. Three seeds measure training variation; they
do not create three independent validation sets. The consistent CNN colour gain
and fine-tuning deterioration exceed the observed paired seed variation, whereas
the smaller MobileNet gains are noisy. Choosing models repeatedly on the same
37 leaves can also favour this particular split. Even the zero camera-like SD
of the frozen combination does not mean its true accuracy is known exactly.

The provisional phone choice is **frozen MobileNet with only the new final layer
trained, 15 epochs, no augmentation**. Selection uses highest mean camera-like
accuracy, then clean accuracy and smaller model size to break ties. Export seed
**0** was fixed in advance, not chosen for the luckiest score. That particular
checkpoint achieved **94.59% clean and 81.08% camera-like accuracy**; the 95.50% /
85.59% figures describe the three-seed setup average.

The 6.10 MB ONNX export includes the four classes, input size 96 and ImageNet
normalization metadata. Its preprocessing uses RGB pixels in NCHW order. ONNX Runtime matched PyTorch over all
37 clean and 37 camera-like inputs: maximum probability error **8.64 × 10⁻⁶**.
The saved JPEG reference error was **4.17 × 10⁻⁷**. The model and reference files
are in `task4_results/phone`; the local export also creates
`ex08_task4_for_agent.zip` as in Task 2e.

**Phone observations still required.** Loading the model into the
[course camera app](https://tomasvicar.github.io/MLR-public/apps/classifier/)
was blocked because browser file-upload permission was declined. No browser
inference or physical phone-camera success is claimed. Load the exported ONNX
file on a phone, try new leaves with different lighting/backgrounds, and replace
this paragraph with the student's actual observations and final model choice.
Submit the student's final `REPORT.md` to `mlr-ex08` within 48 hours of the lab.
