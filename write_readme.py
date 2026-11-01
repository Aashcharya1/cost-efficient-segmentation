"""Keep README results and commands synchronized with the measured submission."""

from src import runtime

runtime.configure()
from pathlib import Path
import json, torch
from src.model import SmallFCN


def main():
    results = {
        s: json.loads(Path("report", s + ".json").read_text())
        for s in ("val", "test", "captured")
    }
    selection = json.loads(
        Path("experiments/compact_selection/results.json").read_text()
    )
    manifest = json.loads(Path("captured/manifest.json").read_text())
    saved = torch.load("params/model_fp16.pt", map_location="cpu", weights_only=True)
    width = saved.get("architecture", {}).get("width_mult", 1.0)
    config = saved["inference"]
    parameters = sum(p.numel() for p in SmallFCN(width_mult=width).parameters())
    table = "| Set | Images | mIoU | BF1 | Q | CPU ms/image | Bytes | CES |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for split, label in [
        ("val", "Validation"),
        ("test", "DUTS-TE"),
        ("captured", "Six own college photos"),
    ]:
        r = results[split]
        table += f'| {label} | {r["n"]} | {r["miou"]:.4f} | {r["bf1"]:.4f} | {r["q"]:.4f} | {r["t_cpu_ms"]:.3f} | {r["s_disk_bytes"]:,} | {r["ces"]:.4f} |\n'
    photo_rows = "\n".join(
        f'| {p["name"]} | {p["object"]} |' for p in manifest["photos"]
    )
    text = f"""# Cost-Efficient Segmentation under a Compute Budget

Track B: a compact FCN trained from scratch on DUTS, with exact assignment metrics, single-thread CPU timing, six user-captured college photos, independent mask annotations, a generated report and a verified submission archive.

## 1. Current submission status

The project contains the trained final model, training/validation evidence, complete DUTS-TE benchmark, six personally captured college photos, masks and overlays. The user confirmed taking the photos around the college in daylight and at night. The old internet substitutes have been archived outside the submitted captured set.

**Source limitation:** the supplied source files came through WhatsApp. Their bytes are preserved unchanged, but unmodified camera originals were not supplied. Mask coordinates were independently traced before viewing predictions; fine bicycle spokes, chair slats and fur boundaries are approximate. Review these masks and obtain camera originals if the instructor interprets "as taken" strictly. No grade or full-marks guarantee is made without the marking rubric.

All six photos are reported; none was removed to increase the test score. Strong examples appear first in the report for presentation only. Official CES is computed on DUTS-TE, not on the captured photos.

## 2. Measured final results

{table}

Selected candidate: **{selection['selected']}**. Width multiplier: {width}; trainable parameters: {parameters:,}. Internal input: {config['input_size']}x{config['input_size']}; flip averaging: {config['flip']}; object-minus-background logit bias: {config['logit_bias']}.

The CPU budget is **50 ms average**, measured over all 5,019 DUTS-TE images. The recorded final average is **{results['test']['t_cpu_ms']:.3f} ms**. The report includes the baseline comparison and all validation-selection evidence, including a timing-variable first optimized run. CPU timing can change with background activity, power state and temperature.

Final checkpoint: `params/model_fp16.pt`. Despite its legacy name, it is a structured mixed-precision parameter file with convolution weights stored as int8 plus per-output-channel scales when selected. It is reconstructed to float32 for CPU execution. No external inference engine is used.

Checkpoint SHA-256: `{results['test']['checkpoint_sha256']}`.

## 3. Files and directories

```text
predict.py                      Required assignment CLI
src/runtime.py                  Environment and one-thread library limits
src/model.py                    Full/half-width FCN encoder-decoder
src/dataset.py                  DUTS pairs, split, augmentation and cache
src/train.py                    Training and validation-only checkpoint selection
src/predict.py                  Complete timed CPU predictor
src/evaluate.py                 Streaming metrics and full-set CPU timing
src/utils.py                    Scratch metrics, CES and checkpoint loading
prepare_captured.py             Resize photos and rasterize manual annotations
optimize_inference.py            Validation-only resolution/threshold/flip grid
optimize_storage.py              Full-width storage comparison on validation
select_compact_model.py          Paired validation selection of final candidate
build_report.py                  Generate report, plots and overlays
write_readme.py                  Synchronize this documentation with results
verify_submission.py             Check evidence, source hashes and real CLI
package_submission.py            Build ZIP and content-hash manifest
finalize.py                      Run frozen evaluations, report, verify and package
requirements*.txt                Supported ranges and tested versions

data/DUTS-TR/                   Training/validation source dataset
data/DUTS-TE/                   Held-out benchmark dataset
data/split.json                 Exact split filenames and dataset fingerprint
data/cache/                     Optional lossless training cache
params/model_fp16.pt             Final inference parameter file
params/training.json             Selected model's training history
params/student/                 Compact-model training artifacts
experiments/baseline/            Original full-width weights and measured results
experiments/compact_selection/   Paired validation protocol and results
experiments/*.json               Inference/storage search records
captured/original/               Six unchanged received WhatsApp JPEGs
captured/images_256/             Corresponding resized PNGs
captured/masks_256/              Independent binary PNG annotations
captured/manifest.json           Source filenames, hashes, targets and provenance
captured/annotations.json        Hand-traced coordinates/primitives
report/REPORT.md                Detailed report
report/figures/                 Every captured overlay, failures and training plot
report/{{val,test,captured}}.json   Final set/per-image measurements
report/submission_sha256.json    Packaged-file hashes
assignment_submission.zip       Final deliverable archive
```

Retired web photos are retained locally in `experiments/web_examples/`, excluded from the submission. DUTS downloads, the optional cache, optimizer resume files, raw candidate weights and logs are also excluded. Baseline weights/results and search JSON evidence are included for audit and reproduction.

## 4. Installation

PowerShell, from the project root:

```powershell
Set-Location F:\\segmentation_project
python --version
python -m pip install -r requirements-lock.txt
```

The tested environment used Python 3.12.1, NumPy 1.26.4, OpenCV 4.11.0.86, PyTorch 2.7.1+cu118 and Matplotlib 3.10.0. The lock file specifies the PyTorch base version. Existing working installations need not be replaced just to predict. CPU-only PyTorch can run inference/evaluation; CUDA was used for training. No external pretrained weights are downloaded.

Optional isolated environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-lock.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Runtime dependencies are limited to Python standard library, PyTorch, NumPy, OpenCV and Matplotlib. No torchvision segmentation models, sklearn, skimage, scipy morphology, library clustering/max-flow/segmentation, OpenCV morphology, ONNX Runtime or separate inference engines are used.

## 5. Quick prediction: no training or dataset download needed

```powershell
python predict.py --image captured/original/body_wash_bottle.jpg --out report/example_mask.png
```

For another input:

```powershell
python predict.py --image "C:\\path\\photo.jpg" --out "C:\\path\\mask.png" --checkpoint params/model_fp16.pt
```

The output is a 256x256 grayscale PNG with background 0 and object 255. The CLI prints exactly two lines:

```text
T CPU ms: <measured average of 20 more runs>
S disk bytes: <actual parameter file byte size>
```

The initial prediction is the untimed warm-up. Twenty additional complete predictions are timed with `time.process_time()`. Loading, original-photo resizing and PNG export are excluded. The CLI's single-image repeat time is separate from the all-DUTS-TE benchmark.

## 6. Data and fixed split

Expected directories:

```text
data/DUTS-TR/DUTS-TR-Image/
data/DUTS-TR/DUTS-TR-Mask/
data/DUTS-TE/DUTS-TE-Image/
data/DUTS-TE/DUTS-TE-Mask/
```

The loader also accepts the uppercase `-MASK` variant. It matches stems across separate image and mask directories, rejects missing masks, and checks the official counts: 10,553 DUTS-TR and 5,019 DUTS-TE pairs.

Download only if data is missing:

```powershell
python download_data.py
# Or use the public archive mirror:
python download_mirror.py
```

Archives are reused and extraction paths are validated. The optional approximately 2.6-GiB uint8 cache accelerates repeated training/validation loading without changing sample tensors:

```powershell
python prepare_cache.py
```

Seed 42, sorted filenames and a 90% floor split give **9,497 training and 1,056 validation images**. Exact names and a fingerprint are saved in `data/split.json`. Keep the manifest; do not regenerate it for a more favourable score. DUTS-TE and the six personal photos are never gradient-training data or checkpoint-selection data.

## 7. Training and reproducibility

The full-width teacher was trained from random initialization for 40 epochs using AdamW (learning rate 0.002, weight decay 0.0001), cosine decay, batch size 32, pixelwise cross entropy plus foreground Soft Dice, horizontal flips and brightness/contrast augmentation. Validation Q selected epoch 31. Source weights/history are in `experiments/baseline/`.

To train a separate full-width reproduction without overwriting delivered weights:

```powershell
python -m src.train --output_dir params/reproduction_teacher --width_mult 1.0 --epochs 40 --batch_size 32 --lr 0.002 --patience 10
```

The compact candidate has half the encoder/decoder channels and starts from random weights. Its supervised loss additionally uses 0.5 times soft-target cross entropy from the original teacher, trained solely within this project. Teacher gradients are disabled; no external pretrained model or extra dataset is used.

```powershell
python -m src.train --output_dir params/student --width_mult 0.5 --distill_checkpoint experiments/baseline/model_fp16.pt --epochs 40 --batch_size 32 --lr 0.002 --patience 8
```

Use a new output directory if preserving the existing compact experiment. Fixed seeds and stored histories support reproduction; GPU numerical results can vary across builds. Resume an interrupted run with the **same** settings/output directory plus `--resume`. This needs its local `resume.pt`, which is not packaged. Increasing the maximum epochs on resume is not a supported schedule extension because the saved scheduler state is restored.

The full-width encoder channels are 16/32/64/96/128 and decoder channels 64/48/32/16; half width uses 8/16/32/48/64 and 32/24/16/8. Internal 160x160 input reduces to 80/40/20/10/5, then skip-connected decoding returns to 80x80 and class logits are upsampled to 256x256. A global pooled context branch augments the bottleneck.

## 8. Validation-only optimization

The original teacher and the grid definition were fixed before the inference search:

```powershell
python optimize_inference.py
python optimize_storage.py
```

The first script tries resolutions 128/160/192/224/256, flip averaging on/off, and seven logit biases. Quality uses all validation images; initial timing estimates repeat a validation photo 64 times. The second script compares float16 storage, per-channel int8 convolution storage, and int8 storage excluding the head/context, using full-validation timing. These commands overwrite the intermediate/default parameter file, so run them in an experimental copy when protecting the final deliverable.

A three-quarter-width candidate (263,786 parameters) is structurally pruned from the project's own teacher using parameter magnitudes, with channel indices propagated consistently through convolutions, normalization, skips and context. It is then fine-tuned for up to ten epochs, still on the training split only:

```powershell
python prune_teacher.py
python -m src.train --output_dir params/pruned --width_mult 0.75 --initialize params/pruned/initial.pt --distill_checkpoint experiments/baseline/model_fp16.pt --epochs 10 --batch_size 32 --lr 0.0005 --patience 5
```

This inherits the teacher's within-project scratch-training lineage, not external pretrained weights. The full-width optimized checkpoint is preserved in `experiments/full_optimized.pt`. After compact training, the final selector compares it against half-width and three-quarter-width float16/int8 variants and a flip-averaged compact candidate:

```powershell
python select_compact_model.py
```

Each candidate processes **every validation photo**, in rotating timing order, to reduce machine-load/order bias. Highest validation CES under 50 ms selects the final architecture/storage/settings. The selector exports `params/model_fp16.pt` and the selected training history. It does not read DUTS-TE or captured photos. Search protocols/results are packaged.

Convolution storage quantization is implemented directly with tensors: scale = max absolute weight per output channel /127; round/clamp to signed int8; store values and float32 scales without additional compression. Same-dtype tensor payloads are concatenated to reduce archive headers without compressing their bytes; reconstruction is bit-identical to the corresponding unpacked checkpoint. Reconstruct float32 weights before timing. TorchScript tracing/freezing/optimization is also prepared before timing. Every per-photo operation remains timed. The regular loader supports both original state dictionaries and structured packed checkpoints.

Baseline DUTS-TE results had already been reported before this improvement pass. New choices were based on validation only; the new DUTS-TE run is explicitly reported as a repeated final benchmark. Do not choose a candidate by its test or personal-photo score.

## 9. Six personal photos and masks

| Basename | Fixed foreground definition |
|---|---|
{photo_rows}

The user confirmed personally taking these around college, in daylight and at night, and that the stall was inside college. Exact per-image lighting, tube lighting and outdoor capture are not claimed. Visible coverage includes near/far object scale, cluttered backgrounds, and partial occlusion of the distant stall chair by foreground merchandise. Image-edge truncation is recorded separately from occlusion.

Each basename appears in all three folders:

```text
captured/original/body_wash_bottle.jpg
captured/images_256/body_wash_bottle.png
captured/masks_256/body_wash_bottle.png
```

WhatsApp sources were renamed without re-encoding. The manifest maps received names to descriptive names and preserves SHA-256 hashes. Photos are resized with `cv2.INTER_AREA`. The masks were drawn by coordinate tracing on the 256x256 copies before inspecting predictions, not derived from model outputs. Background holes, shadows and non-target objects are excluded according to the documented definitions. Bicycle spokes/tyres and other fine structures remain approximate.

Reproduce the resized photos and masks:

```powershell
python prepare_captured.py
```

This reads the six unchanged sources and `captured/annotations.json`; it uses a hand-written polygon/line/ellipse rasterizer, not a segmentation or morphology library. If you manually improve the masks, also update the source annotations/rasterizer or keep reviewed masks separately so this command does not overwrite those improvements. Inspect each overlay and check source provenance before submission.

## 10. Exact metrics, timing and CES

- **mIoU:** accumulate one 2x2 confusion matrix over the whole set, calculate background/object IoU and average. It is not the mean of per-image IoUs. A zero-union class has IoU zero; neither DUTS aggregate class is absent.
- **Boundary F1:** object pixels touching four-connected background are boundaries; outside pixels are background. Match within a 5x5 square window, compute per-image F1 and average. Neither boundary gives 1; only one gives 0. Window matching is implemented directly with NumPy.
- **Quality:** Q = 0.70 mIoU +0.30 BF1.
- **Score:** CES = Q [1 -0.20 T_CPU/50 -0.10 min(1,S_disk/4194304)] if T_CPU<=50 ms; otherwise zero. Time and size penalties are added inside the bracket.

Before numerical imports, runtime setup sets OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and VECLIB_MAXIMUM_THREADS to 1. OpenCV and PyTorch intra/inter-op threads are also 1. Timed inference starts from the in-memory 256x256 uint8 BGR photo and ends with a uint8 0/1 mask. Colour conversion, tensor creation, normalization, internal resizing, all model calls, upsampling and thresholding count. Reading/writing, initial original-photo resizing, parameter reconstruction/loading and TorchScript preparation do not count.

Full evaluation warms up on a real photo, uses `time.process_time()` for each individual prediction and averages all images. The predictor uses no subprocesses or worker processes. Orchestration/verification may launch scripts outside the timed function. Single-core timing is hardware/load dependent; the report retains the observed timing-variable first full-width optimized test run.

## 11. Evaluate, report and package

Only run after model selection is finished and the checkpoint is frozen:

```powershell
python -m src.evaluate --split val
python -m src.evaluate --split test
python -m src.evaluate --split captured
python build_report.py
python verify_submission.py
python write_readme.py
python package_submission.py
```

Or run:

```powershell
python finalize.py
```

This overwrites evaluation JSON, generated figures/report, README, manifest and ZIP. Make durable narrative changes in the generators. `src.evaluate` accepts `--data_root`, `--checkpoint`, `--captured_root` and `--output`; downstream generators/verification use the documented default project paths.

All final set records must reference the same checkpoint hash. The report includes all six captured overlays, two actual DUTS-TE failures, CPU model, splits, training details, optimization evidence and limitations. It orders captured examples by Q for presentation but does not change the evaluation set.

## 12. Tests and submission checks

```powershell
python -m unittest discover -s tests -v
python verify_submission.py
```

Tests cover the independent brute-force boundary oracle, empty/full/border cases, global-confusion mIoU, the CES formula, legacy weights, packed weights, architecture reconstruction, TorchScript configuration and mask shape/values. Verification checks 5–10 captured pairs, exact source hashes, all three evaluation counts, shared checkpoint identity, aggregate/per-image consistency, the CPU budget and the real CLI's exact two-line stdout.

Before uploading:

1. Review every annotation and overlay, especially the bicycle, distant chair and furry edges.
2. Add your student name/roll number/course details if requested.
3. Keep `report/figures/` beside the Markdown report, or export the report to your instructor's required format.
4. If required, supply camera originals rather than WhatsApp copies. Their absence is disclosed, not hidden.
5. Extract the ZIP into a fresh folder and predict one included photo. Inference must not need the DUTS dataset or training cache.
6. Submit the rebuilt `assignment_submission.zip` and any separately requested report.

The ZIP stores the final parameter file with ZIP_STORED, avoiding additional compression. A SHA-256 manifest covers packaged files. Raw training data/cache, optimizer state and retired web examples are omitted.

## 13. Troubleshooting and demonstration

| Issue | Check |
|---|---|
| Missing dataset or wrong count | Extraction structure and matching masks; do not bypass count checks by dropping samples. |
| Dataset fingerprint/cache mismatch | Restore the matching split/data or deliberately rebuild the cache; preserve experiment provenance. |
| CUDA unavailable | Inference/evaluation remain CPU-only; training falls back to CPU. |
| GPU out of memory during a new run | Reduce batch size and record the change; preserve the delivered checkpoint. |
| Missing captured pair | Match PNG stems across images_256 and masks_256; keep the manifest consistent. |
| Stale results/hash error | Stop training, freeze one file, then regenerate all three evaluation records. |
| Report text reverts | Edit the generator, not only generated Markdown. |
| Timing varies | Report actual full-set CPU measurements; background activity and power/thermals affect clock speed. Do not substitute a favourable single-image repeat. |
| Poor segmentation | Discuss errors honestly; never adjust true masks to match predictions. |

For a viva, understand encoder/decoder/skip/context operations, depthwise refinement, CE/Dice and distillation, split roles, global mIoU versus per-image BF1, exact timing boundaries, storage quantization versus computation precision, TorchScript preparation, CES penalties and the actual failure examples. The model meets a compute budget; limited semantic capacity, clutter, ambiguity and thin structures can still reduce quality.

## 14. References

Assignment specification supplied by the user; Wang et al., *Learning to Detect Salient Objects with Image-level Supervision*, CVPR 2017, [DUTS](https://saliencydetection.net/duts/); [archive mirror](https://huggingface.co/datasets/chitradrishti/duts); user-provided [implementation reference](https://github.com/damarasingu/Cost-Efficient-Segmentation). That reference's weights, photos, masks and scores were not reused. The final captured set consists of the user's college photos, not web substitutes.
"""
    text += """

## Final report and submission

- [Final ten-page report](Image_segmentation.pdf)
- [Complete submission ZIP](assignment_submission.zip)
- [Exact submission checklist](SUBMISSION.md)
- [GitHub repository](https://github.com/Aashcharya1/cost-efficient-segmentation-cv-assignment)

The repository is private. Invite the instructor/TA if they need repository access; sharing its URL alone is insufficient. The ZIP includes the supplied report PDF, code, final weights, six source images and masks, and evaluation evidence. DUTS downloads, the optional cache, optimizer resume states and retired web examples are excluded.

The supplied PDF is preserved unchanged and is not regenerated by the Markdown report builder. Replace the root PDF before rerunning package_submission.py if you export a revised version.
"""
    Path("README.md").write_text(text, encoding="utf-8")
    print("Updated README.md")


if __name__ == "__main__":
    main()
