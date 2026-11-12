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
