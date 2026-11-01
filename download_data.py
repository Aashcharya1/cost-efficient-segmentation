"""Download DUTS from its official site, or use --mirror for a public archive mirror."""

import argparse
from pathlib import Path
import time
import urllib.request
import zipfile


def prepare(root="data", mirror=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for split in ("TR", "TE"):
        name = f"DUTS-{split}"
        suffix = "-mirror" if mirror else ""
        destination = root / f"{name}{suffix}.zip"
        if not destination.exists() or not zipfile.is_zipfile(destination):
            base = (
                "https://huggingface.co/datasets/chitradrishti/duts/resolve/main"
                if mirror
                else "https://saliencydetection.net/duts/download"
            )
            url = f"{base}/{name}.zip"
            print(f"Downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            temporary = destination.with_suffix(".partial")
            with urllib.request.urlopen(
                request, timeout=60
            ) as response, temporary.open("wb") as output:
                total = 0
                last = time.monotonic()
                expected = int(response.headers.get("Content-Length", 0))
                while chunk := response.read(2**20):
                    output.write(chunk)
                    total += len(chunk)
                    if time.monotonic() - last > 15:
                        print(f"{name}: {total / 2**20:.1f} MiB", flush=True)
                        last = time.monotonic()
            if expected and total != expected:
                raise IOError(f"Incomplete download: {total} of {expected} bytes")
            if not zipfile.is_zipfile(temporary):
                raise ValueError(f"Not a ZIP archive: {url}")
            temporary.replace(destination)
        else:
            print(f"Reusing {destination}", flush=True)
        with zipfile.ZipFile(destination) as archive:
            for info in archive.infolist():
                if not (root / info.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe archive path: {info.filename}")
            archive.extractall(root)
        print(f"Ready: {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--mirror", action="store_true")
    args = parser.parse_args()
    prepare(args.data_root, args.mirror)


if __name__ == "__main__":
    main()
