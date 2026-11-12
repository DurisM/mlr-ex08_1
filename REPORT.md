# Task 4 - Better Leaf Classifier

## What I did

I tried to improve my leaf classifier from Task 3. I tested color augmentation, MobileNet, more epochs and different training settings.

I used the same 136 training and 37 validation images for all models. Each model was trained 3 times with different seeds (0, 1, 2). I also tested on camera-like images with changed colors, rotation and cropping.

## Results

Results are averages from 3 runs.

| Model                               | Validation | Camera-like |
| ----------------------------------- | ---------: | ----------: |
| CNN baseline                        |     66.67% |      57.66% |
| CNN no geometry                     |     83.78% |      57.66% |
| CNN + color                         |     69.37% |      72.07% |
| MobileNet frozen                    |     95.50% |      85.59% |
| MobileNet fine-tuned                |     71.17% |      62.16% |
| Fine-tuned + geometry               |     81.98% |      73.87% |
| Fine-tuned + geometry + color       |     77.48% |      78.38% |
| Fine-tuned + cosine                 |     75.68% |      72.97% |
| Fine-tuned, 30 epochs               |     81.98% |      68.47% |
| Fine-tuned + all changes, 30 epochs |     88.29% |      85.59% |
| Frozen + all changes, 30 epochs     |     84.68% |      83.78% |

## What I found

Color augmentation helped the CNN a lot, camera-like accuracy improved by around 14%. Removing geometry improved normal accuracy but didnt help camera-like results.

Frozen MobileNet was the best. I expected training the whole MobileNet to be better, but it was actually worse. More epochs and cosine scheduling helped sometimes, but results changed between seeds.

Since there are only 37 validation images, one image changes accuracy by 2.7%. So small differences might just be random. But the color improvement and worse fine-tuning happened in all 3 runs.

## My choice

I would use **frozen MobileNet**, training only the last layer. It got 95.50% validation and 85.59% camera-like accuracy. Its also simpler than the other versions.

I exported it to ONNX, but couldnt finish the real phone test because of browser upload permissions. So I dont know yet how good it works on actual phone camera photos.
