# Leaf Lens — MLR exercise 08

**Open the app: https://durism.github.io/mlr-ex08/**

A buildless leaf classifier: one `index.html` containing plain JavaScript, CSS and an inline QR code. The supplied ONNX model and JSON/image assets sit alongside it. Hosted with GitHub Pages from `main`, repository root.

## Use

Open the HTTPS link in Safari on iPhone, tap **Start rear camera**, and allow camera access. Center a well-lit leaf in the square. The probabilities for apple, cherry, chestnut and maple update continuously. **Stop camera** releases the camera; leaving the page also stops it. Tap Start to resume after returning.

Use **Upload a photo** if camera permission is denied or a camera is unavailable. Images are processed on the device and are never uploaded. An internet connection is needed to load the model and CDN libraries.

## Exact model preprocessing

`model_info.json` is read at runtime; class order and preprocessing settings are not duplicated in JavaScript:

1. Crop the central square: side = `min(width, height)`, origin = `((width-side)/2, (height-side)/2)`.
2. Resize to `input_size` × `input_size` (96 × 96 for the supplied model), using the browser canvas with high-quality smoothing.
3. Extract RGB, divide each byte by 255, then apply `(channel - mean[channel]) / std[channel]`.
4. Pack float32 values in NCHW order, shape `[1, 3, input_size, input_size]`.
5. Run the model using its declared input/output names and apply stable softmax to its four output logits.

Mean: `[0.485, 0.456, 0.406]`; standard deviation: `[0.229, 0.224, 0.225]`.

ONNX Runtime Web 1.22.0 loads from jsDelivr and uses single-threaded WebAssembly, supported by Safari without cross-origin isolation headers. The video uses `autoplay`, `muted` and `playsinline`, starts with a user gesture, and requests `facingMode: environment`. Only one prediction runs at a time.

JPEGs, including uploaded photos, are decoded locally with the pinned jSquash/MozJPEG decoder (1.6.0, loaded from jsDelivr). Safari's native JPEG decoder produced slightly different pixels for the reference image; consistent decoding resolves that difference. Other image formats use native browser decoding. No framework or build step is required.

## Reference verification

Tap **Run reference photo check** to run `check_image.jpg` through the same preprocessing and model path used for photos and camera frames, then compare every class with `check_expected.json`. PASS requires maximum absolute probability error ≤ `0.00001` (to allow floating-point differences).

Verified in WebKit with an iPhone-sized viewport and in Chromium:

| Class | Expected probability | Browser probability |
| --- | ---: | ---: |
| apple | 0.9672254324 | 0.9672252840 |
| cherry | 0.0249426793 | 0.0249426902 |
| chestnut | 0.0074095908 | 0.0074096636 |
| maple | 0.0004223514 | 0.0004223623 |

Maximum absolute error: `1.484 × 10^-7`. Photo upload and denied-camera fallback were checked in WebKit. Central-square crop, channel normalization, NCHW layout, repeated inference and camera track cleanup were checked with a simulated stream. Physical iPhone camera hardware was not available for testing.

## QR code

The labelled QR code at the bottom of the app is an inline SVG, generated locally with Python's `qrcode` package. It encodes **https://durism.github.io/mlr-ex08/** and has a clickable text link alongside it. No external QR-image service is used. Its encoded URL was independently decoded and compared with the app's public URL.

## Local preview

Serve this directory over HTTP, for example `python3 -m http.server 8000`, then open `http://localhost:8000`. Phone camera access requires an HTTPS deployment rather than an ordinary HTTP LAN address.

## Libraries

- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), MIT license.
- [jSquash JPEG](https://github.com/jamsinclair/jSquash), Apache-2.0 license; MozJPEG decoder included by that package.

The original model, description and check files are supplied course assets.

## Task 4 — controlled comparisons

The Task 4 work keeps the existing Task 3 `LeafDataset`, `LeafNetBN` and batch
training loop. The only tested changes are colour augmentation, frozen versus
fine-tuned pretrained MobileNet, geometric/colour augmentation, cosine scheduling
and more epochs. A CNN without geometry is included as a control.

- [REPORT.md](REPORT.md): measured results draft; the student must write/revise the
  final interpretation and add real phone-camera observations.
- [task4.py](task4.py): readable training, comparison and export code.
- [Notebook](Copy_of_ex08_student.ipynb): original exercise with focused Task 4 cells
  and the corrected group dataset link.
- [All seed measurements](task4_results/results.csv),
  [paired comparisons](task4_results/comparison_table.md), and per-run JSON files.
- [Phone model](task4_results/phone/leaf_model.onnx) and
  [export verification](task4_results/phone/export_validation.json).

Use Python 3.11 and install `task4_requirements.txt`. Download/extract the supplied
class ZIP into `data/leaves` using Task 2a of the notebook, then run:

```sh
python task4.py compare
python task4.py combine
python task4.py export
```

The saved combination plan explains why those combinations were chosen after the
individual rows. Every comparison uses the same source images and seeds 0, 1, 2.
Camera-like validation is materialized once with seed 2026. Results are final-epoch
weights; selection for export uses mean camera-like accuracy, then clean accuracy
and model size as tie-breakers. Seed 0 is fixed in advance for the exported model.

The [course camera app](https://tomasvicar.github.io/MLR-public/apps/classifier/)
can load the exported ONNX file directly. Local ONNX predictions are checked
against PyTorch on all 37 clean and 37 camera-like inputs. The course-app upload
was blocked by declined browser file permission; a physical phone trial remains
for the student. This separate Task 4 export does not change the existing Leaf
Lens app's model or its earlier reference check.

Raw datasets, transformed validation images, training checkpoints and dependency
caches are not committed. The code regenerates them from the supplied split.
