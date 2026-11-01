"""Generate the measured report, personal-photo overlays and frozen-model comparisons."""

from src import runtime
import json, platform, hashlib
from pathlib import Path
import cv2, numpy as np, torch
from src.predict import Predictor
from src.model import SmallFCN
from src.utils import Metrics


def main():
    runtime.configure()
    root = Path("report")
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = {
        s: json.loads((root / (s + ".json")).read_text())
        for s in ("val", "test", "captured")
    }
    training = json.loads(Path("params/training.json").read_text())
    manifest = json.loads(Path("captured/manifest.json").read_text())
    selected = json.loads(Path("experiments/selected_inference.json").read_text())
    compact = json.loads(Path("experiments/compact_selection/results.json").read_text())
    storage = json.loads(Path("experiments/storage_search.json").read_text())
    baseline = {
        s: json.loads(Path("experiments/baseline", s + ".json").read_text())
        for s in ("val", "test")
    }
    checkpoint = Path("params/model_fp16.pt")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert all(
        r["checkpoint_sha256"] == digest for r in results.values()
    ), "Stale evaluation results"
    predictor = Predictor(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    width = saved.get("architecture", {}).get("width_mult", 1.0)
    parameter_count = sum(p.numel() for p in SmallFCN(width_mult=width).parameters())
    config = saved["inference"]
    encoder = ", ".join(str(int(v * width)) for v in (16, 32, 64, 96, 128))
    decoder = ", ".join(str(int(v * width)) for v in (64, 48, 32, 16))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = training["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, label in [("miou", "mIoU"), ("bf1", "Boundary F1"), ("q", "Q")]:
        axes[0].plot(
            [r["epoch"] for r in history], [r[key] for r in history], label=label
        )
    axes[0].set(xlabel="Epoch", ylabel="Validation score", ylim=(0, 1))
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot([r["epoch"] for r in history], [r["loss"] for r in history])
    axes[1].set(xlabel="Selected training-phase epoch", ylabel="Training objective")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "training.png", dpi=160)
    plt.close(fig)

    def composite(im, gt, pred):
        panels = [im]
        for mask, color in [(gt, (45, 210, 45)), (pred, (240, 180, 20))]:
            out = im.copy()
            out[mask] = (im[mask] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
            panels.append(out)
        return np.concatenate(panels, axis=1)

    overlays = []
    capture_scores = []
    capture_ranking = {
        Path(r["image"]).stem: r["q"] for r in results["captured"]["per_image"]
    }
    for photo in sorted(
        manifest["photos"], key=lambda p: capture_ranking[p["name"]], reverse=True
    ):
        name = photo["name"]
        im = cv2.imread(f"captured/images_256/{name}.png")
        gt = cv2.imread(f"captured/masks_256/{name}.png", 0) >= 128
        pred = predictor(im) > 0
        cv2.imwrite(str(figures / (name + ".png")), composite(im, gt, pred))
        meter = Metrics()
        meter.update(pred.astype(np.uint8), gt.astype(np.uint8))
        r = meter.result()
        capture_scores.append((name, r))
        overlays.append(
            f'### {name.replace("_"," ").title()}\n\nTarget: {photo["object"]}.\n\nInput | true-mask overlay (green) | predicted-mask overlay (cyan).\n\n![{name}](figures/{name}.png)\n\nmIoU {r["miou"]:.4f}; BF1 {r["bf1"]:.4f}; Q {r["q"]:.4f}.\n'
        )
    worst = sorted(results["test"]["per_image"], key=lambda r: r["q"])[:2]
    failures = []
    analysis_path = root / "failure_analysis.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    for i, record in enumerate(worst, 1):
        ip = Path("data/DUTS-TE/DUTS-TE-Image") / record["image"]
        masks = list(Path("data/DUTS-TE").glob(f"*Mask/{ip.stem}.*")) + list(
            Path("data/DUTS-TE").glob(f"*MASK/{ip.stem}.*")
        )
        im = cv2.resize(cv2.imread(str(ip)), (256, 256), interpolation=cv2.INTER_AREA)
        gt = (
            cv2.resize(
                cv2.imread(str(masks[0]), 0),
                (256, 256),
                interpolation=cv2.INTER_NEAREST,
            )
            >= 128
        )
        pred = predictor(im) > 0
        cv2.imwrite(str(figures / f"failure_{i}.png"), composite(im, gt, pred))
        fp = int((pred & ~gt).sum())
        fn = int((gt & ~pred).sum())
        discussion = analysis.get(
            record["image"],
            "Foreground-selection and contour errors are visible in the overlays. Limited semantic context and the small internal feature maps are plausible contributors; inspect the actual image to distinguish these hypotheses.",
        )
        failures.append(
            f'### Failure {i}: {record["image"]}\n\n![Failure {i}](figures/failure_{i}.png)\n\nmIoU {record["miou"]:.4f}, BF1 {record["bf1"]:.4f}, Q {record["q"]:.4f}; false-positive pixels {fp:,}, false-negative pixels {fn:,}.\n\n{discussion}\n'
        )
    table = "| Set | Images | mIoU | BF1 | Q | CPU ms/image | Parameter bytes | CES |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for split, label in [
        ("val", "Validation"),
        ("test", "DUTS-TE"),
        ("captured", "Own college photos"),
    ]:
        r = results[split]
        table += f'| {label} | {r["n"]} | {r["miou"]:.4f} | {r["bf1"]:.4f} | {r["q"]:.4f} | {r["t_cpu_ms"]:.3f} | {r["s_disk_bytes"]:,} | {r["ces"]:.4f} |\n'
    candidates = "| Storage / implementation | Validation Q | Full-validation CPU ms | Bytes | Validation CES |\n|---|---:|---:|---:|---:|\n"
    b = baseline["val"]
    candidates += f'| Baseline eager FP16 storage | {b["q"]:.6f} | {b["t_cpu_ms"]:.3f} | {b["s_disk_bytes"]:,} | {b["ces"]:.6f} |\n'
    for r in storage["records"]:
        candidates += f'| TorchScript / {r["storage"]} | {r["q"]:.6f} | {r["t_cpu_ms"]:.3f} | {r["s_disk_bytes"]:,} | {r["ces"]:.6f} |\n'
    compact_table = "| Candidate | Validation Q | Paired CPU ms | Bytes | Validation CES |\n|---|---:|---:|---:|---:|\n"
    for r in compact["records"]:
        compact_table += f'| {r["candidate"]} | {r["q"]:.6f} | {r["t_cpu_ms"]:.3f} | {r["s_disk_bytes"]:,} | {r["ces"]:.6f} |\n'
    best = max(history, key=lambda r: r["q"])
    new = results["test"]
    old = baseline["test"]
    delta = new["ces"] - old["ces"]
    sources = "\n".join(
        f'| {p["name"]} | {p["original_filename"]} | {", ".join(p["conditions_observed"])} |'
        for p in manifest["photos"]
    )
    text = f"""# Cost-Efficient Segmentation under a Compute Budget

Track B: compact fully convolutional segmentation trained from scratch.

## Final results

{table}

The full DUTS-TE mean CPU time is **{new['t_cpu_ms']:.3f} ms**, below 50 ms. Compared with the recorded baseline, DUTS-TE CES changes from **{old['ces']:.6f} to {new['ces']:.6f}** ({delta:+.6f}, {delta/old['ces']*100:+.2f}% relative). Baseline and final timings were measured in separate runs and may vary with the laptop's power/thermal state. The personal-photo row is a test-only generalisation result, not the common CES benchmark.

## Method and training

The selected width multiplier is {width}; encoder channels are {encoder}. Each stage combines a standard 3x3 convolution with batch normalization/ReLU and depthwise 3x3 plus pointwise 1x1 refinement. Decoder channels are {decoder}, with concatenated encoder skip features. A projected global-average-pooled context vector is added at the bottleneck. A two-logit head and bilinear upsampling restore 256x256 predictions. There are {parameter_count:,} trained parameters.

The assignment input photo is resized to 256x256 with OpenCV INTER_AREA. Masks use INTER_NEAREST and threshold >=128. The network internally downsizes its tensor to {config['input_size']}x{config['input_size']}; this is included in timing. The final decision is object logit minus background logit > {config['logit_bias']}. Horizontal-flip averaging is {"enabled" if config['flip'] else "disabled"}.

DUTS-TR has 10,553 pairs. A fixed filename-based seed-42 split gives **9,497 training and 1,056 validation images**, with no overlap. DUTS-TE has **5,019** pairs. The split/fingerprint is in `data/split.json`. No DUTS-TE or captured image was used for gradient training, threshold selection, resolution selection, checkpoint selection or early stopping.

The selected model's recorded training phase ran for {len(history)} epochs; phase epoch {best['epoch']} had its best validation Q. The original teacher was trained from scratch for 40 epochs and selected at epoch 31; pruning/fine-tuning phases inherit that training lineage, rather than representing the entire training history on their own. AdamW used initial learning rate 0.002, weight decay 0.0001 and cosine decay, batch size 32, random horizontal flips and brightness/contrast augmentation. Loss was pixelwise cross entropy plus foreground Soft Dice. CUDA mixed precision was training-only. A lossless uint8 training/validation cache accelerated loading; it does not enter the timed predictor. GPU training seeds were fixed, but exact numerical reproduction may vary across software/hardware builds.

A half-width candidate with 118,578 parameters was additionally trained from random initialization for up to 40 epochs. Alongside true-mask CE + Dice, it used 0.5 times a soft-target cross entropy from the original full-width teacher, itself trained from scratch solely on this project's DUTS-TR training split. No external pretrained weights or additional data were used. Teacher gradients were disabled. Validation Q selected its checkpoint; personal photos and DUTS-TE never contributed gradients or candidate selection. A third candidate used structured channel pruning to three-quarter width (263,786 parameters), initialized solely from this project's own teacher weights, followed by up to 10 fine-tuning epochs at learning rate 0.0005 with the same training/validation roles and teacher loss. Hidden/output channels were ranked by convolution magnitude times batch-normalization scale, with indices propagated through depthwise layers, skips and context/head connections. No external pretrained weights were involved. Original model selection was followed by validation-only inference/storage selection. The baseline DUTS-TE results had already been reported before this pass; therefore this is a repeated final benchmark, not a claim that DUTS-TE has never previously been inspected. The new grid was fixed in advance and the search scripts read validation data only. The final model was frozen before its new DUTS-TE run.

![Training history](figures/training.png)

## Improving CES using validation only

The recorded grid tried internal sizes 128, 160, 192, 224 and 256; flip averaging on/off; and logit biases -0.75, -0.5, -0.25, 0, 0.25, 0.5 and 0.75. Every quality measurement used all 1,056 validation images. Initial grid CPU estimates used 64 repetitions after warm-up on a validation photo. Selection used estimated validation CES, never DUTS-TE or captured scores. The initial full-width grid retained size {selected['config']['input_size']}, flip={selected['config']['flip']}, bias={selected['config']['logit_bias']}. The later compact-model comparison also tested flip averaging on its packed half-width candidate.

Allowed TorchScript tracing, freezing and `optimize_for_inference` fold/reorder inference operations and remove Python/module overhead. Compilation and parameter loading happen before timing; every per-photo operation remains timed. Convolution kernels are then stored as signed 8-bit values with one float32 scale per output channel, without gzip/ZIP compression of their tensor payloads. Biases and floating-point normalization state retain float16 storage. Same-dtype tensors are concatenated into raw arrays to reduce per-tensor archive headers; no tensor payload is compressed. Values are reconstructed to a float32 PyTorch model before timing. **This is storage quantization, not a separate inference engine.** All quantized values, scales and validated inference settings are contained in the single saved parameter file.

Three storage alternatives were compared using full one-photo-at-a-time validation CPU passes:

{candidates}

These measurements selected the full-width storage candidate. The final architecture/storage selection then compared it with half-width and three-quarter-width variants using a **paired full-validation pass**: all candidates saw every validation photo, with candidate timing order rotated for each image to reduce background-load/order bias.

{compact_table}

Selected candidate: **{compact['selected']}**. The highest paired validation CES selected the final model and storage format. The tiny validation quality change from storage rounding was outweighed by the smaller disk penalty. `experiments/validation_search.json` and `experiments/storage_search.json` contain the evidence. The legacy filename `params/model_fp16.pt` now contains a structured mixed-precision checkpoint, not exclusively float16 tensors. The ordinary loader remains backward compatible with the original state dictionary.

## Timing and exact metrics

CPU: **AMD Ryzen 7 4800H with Radeon Graphics**. Training GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB. Python {platform.python_version()}, PyTorch {torch.__version__}, NumPy {np.__version__}, OpenCV {cv2.__version__}.

OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and VECLIB_MAXIMUM_THREADS are set to 1 before numerical imports; OpenCV, PyTorch intra-op and inter-op thread limits are 1. The timed predictor uses no subprocesses or worker processes. After a real-photo warm-up, `time.process_time()` measures every image individually, beginning with its 256x256 uint8 BGR photo in memory and ending with a 256x256 uint8 0/1 mask. Colour conversion, normalization, tensor creation, internal downscaling, network execution, output upscaling and thresholding count. Disk I/O, the initial original-photo resize to 256x256, loading/reconstructing parameters and TorchScript preparation do not count. DUTS-TE time is summed over all 5,019 images and divided by 5,019. The required CLI separately averages 20 more predictions after its initial prediction.

mIoU comes from one accumulated 2x2 confusion matrix over the full set, averaging object and background IoU. A zero-union class is assigned zero; both classes occur in the DUTS aggregate. Boundaries are foreground pixels touching four-connected background, with outside pixels treated as background. A hand-written separable logical 5x5 window matches edges. BF1 is computed per image and averaged; neither boundary present gives 1, one absent gives 0. Independent brute-force oracle tests verify these definitions.

Q = 0.70 mIoU + 0.30 BF1. CES = Q [1 - 0.20 T_CPU/50 - 0.10 min(1, S_disk/4194304)] if T_CPU <=50 ms, otherwise zero. S_disk is the actual single parameter file's byte size. The delivery ZIP stores it without further compression. Final SHA-256: `{digest}`. All three final evaluation JSON files reference this hash.

## Six personally captured college photos

The user confirmed: "I took these around the college, under daylight and night; the stall was inside college." The six WhatsApp JPEGs were renamed descriptively and preserved **byte-for-byte** as received. Original WhatsApp filenames and hashes are recorded in `captured/manifest.json`. Camera originals and full capture metadata were not supplied; `original/` contains the received WhatsApp sources, which may have been recompressed before receipt. Exact per-image lighting, tube-light illumination and outdoor capture are not asserted.

| Name | Received filename | Visible conditions / relative distance |
|---|---|---|
{sources}

The set covers **near/far relative object scale**, **cluttered backgrounds**, and **partial occlusion of the stall chair by foreground merchandise**, providing three of the assignment's five condition categories. The stall is particularly ambiguous because several objects compete; its central chair was specified as the target before predictions were inspected. No photos were discarded because of model performance. Image-edge truncation, such as the cat's cropped body, is recorded separately and is not counted as occlusion.

Masks were independently traced on the resized 256x256 photos before viewing model outputs. They do not use this model's predictions as ground truth. The central chair, bottle, bicycle, giraffe toy, complete turtle lamp and visible cat are the explicit targets. Background holes are excluded; bicycle tyres/spokes use manually positioned geometric approximations. These masks are approximate, especially for spokes, chair slats and furry edges, and should receive a final human review. Source-coordinate annotations and a reproducible from-scratch rasterizer are included. All six pairs are test-only.

## True and predicted mask overlays

All six photos are included, ordered by final per-image Q for presentation only. For each image: original, green true-mask overlay, cyan prediction overlay. Green masks are the independently prepared annotations, not authoritative external labels.

{chr(10).join(overlays)}

## Personal-photo failure analysis

The bicycle prediction fills much of the open wheel/frame space instead of preserving the visible tyre/spoke structure. The low-resolution representation and bilinear output reconstruction favour broad regions over thin topology. Its manually approximated spokes also make the exact BF1 uncertain. The distant stall chair competes with other chairs and merchandise; the prediction selects nearby distracting regions and misses much of the specified target. Small object scale, clutter and partial occlusion make this a challenging saliency decision. These are actual final failures, not grounds for removing photos from the evaluation.

The body-wash bottle is the strongest example, with an almost complete silhouette; small hand/finger false positives remain. The lamp also localizes well. All six individual metrics and overlays are retained above.

## Failure cases and limitations

The two lowest-Q DUTS-TE images below were selected for post-hoc explanation only, after freezing the model. Input, true overlay and predicted overlay appear left to right.

{chr(10).join(failures)}

The saved model's validation Q is {results['val']['q']:.4f}, compared with test Q {new['q']:.4f}; the generalisation gap remains. A compact network trained from scratch has limited semantic understanding. Its initial internal resize and five reductions can lose small structures, while clutter and competing large regions can cause the wrong salient object to be selected. Skip connections restore spatial cues but cannot recreate all detail lost in downsampling. Quantized storage improves cost efficiency without solving these semantic limitations. Approximate personal annotations add uncertainty to their BF1 values. The original full-width packed candidate initially measured CES 0.589629 on DUTS-TE because its CPU time rose during that run (9.9 ms early versus roughly 14–19 ms later). That run is preserved in `experiments/full_optimized_test_first_run.json`; it was not discarded from the audit record. Paired validation timing, rather than that test result, selected the final architecture. The six-photo timing estimate is less stable than the 5,019-image benchmark.

## Reproduction and submission

See README for setup, prediction, training, validation-only searches, frozen evaluation, mask reproduction and packaging. The report generator reads current JSON/manifest files and verifies checkpoint identity. `verify_submission.py` checks image/mask pairing, source-file preservation, metrics and CLI output. `package_submission.py` builds the archive and SHA-256 manifest, excluding DUTS downloads, cache, optimizer resume state and retired web examples. The baseline weights/results and validation search evidence are included for audit and reproduction; inference uses only the final parameter file.

The implementation, full-set benchmark, six user-confirmed personal photos, independently traced masks and overlays are provided. Strict demands for original camera files rather than WhatsApp copies, mask precision, capture-condition interpretation and the final grade remain matters for the instructor/user to verify; no full-marks guarantee is made.

References: assignment supplied by the user; Wang et al., Learning to Detect Salient Objects with Image-level Supervision, CVPR 2017, https://saliencydetection.net/duts/ ; public archive mirror https://huggingface.co/datasets/chitradrishti/duts ; user-provided implementation reference https://github.com/damarasingu/Cost-Efficient-Segmentation (its weights, photos and scores were not reused).
"""
    (root / "REPORT.md").write_text(text, encoding="utf-8")
    print("Wrote report/REPORT.md")


if __name__ == "__main__":
    main()
