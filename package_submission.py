"""Package deliverables; parameter bytes are stored without outer compression."""

from pathlib import Path
import hashlib, json, zipfile

root = Path(__file__).resolve().parent
files = []
for pattern in (
    "*.py",
    "README.md",
    "SUBMISSION.md",
    "Image_segmentation.pdf",
    ".gitignore",
    ".gitattributes",
    "requirements*.txt",
    "src/*.py",
    "tests/*.py",
    "captured/**/*.jpg",
    "captured/**/*.png",
    "captured/*.json",
    "report/**/*.md",
    "report/**/*.png",
    "report/*.json",
    "params/model_fp16.pt",
    "params/training.json",
    "data/split.json",
    "experiments/predeclared_grid.json",
    "experiments/validation_search.json",
    "experiments/selected_inference.json",
    "experiments/storage_search.json",
    "experiments/selected_storage.json",
    "experiments/baseline/val.json",
    "experiments/baseline/test.json",
    "experiments/baseline/training.json",
    "experiments/baseline/model_fp16.pt",
    "experiments/full_optimized.pt",
    "experiments/full_optimized_test_first_run.json",
    "experiments/compact_selection/*.json",
    "params/student/training.json",
    "params/pruned/training.json",
):
    files.extend(p for p in root.glob(pattern) if p.is_file())
files = sorted(
    p
    for p in set(files)
    if p.name
    not in {
        "bench_training.py",
        "early_smoke_prediction.png",
        "web_contact_sheet.png",
        "annotation_review.png",
        "submission_sha256.json",
    }
)
manifest = {
    str(p.relative_to(root))
    .replace("\\", "/"): hashlib.sha256(p.read_bytes())
    .hexdigest()
    for p in files
}
(root / "report/submission_sha256.json").write_text(json.dumps(manifest, indent=2))
files.append(root / "report/submission_sha256.json")
with zipfile.ZipFile(
    root / "assignment_submission.zip", "w", compression=zipfile.ZIP_STORED
) as z:
    for p in files:
        z.write(p, p.relative_to(root))
print(
    "Packaged",
    len(files),
    "files; bytes",
    (root / "assignment_submission.zip").stat().st_size,
)
