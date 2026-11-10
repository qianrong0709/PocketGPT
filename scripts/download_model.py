"""Download one versioned PocketGPT GitHub Release asset and verify its hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen


MANIFEST = Path(__file__).resolve().parents[1] / "models" / "pocketgpt-v1.json"


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(output="weights/pocketgpt-v1.pt", verify_only=False, manifest_file=MANIFEST):
    manifest = json.loads(Path(manifest_file).read_text())
    prefix = "https://github.com/qianrong0709/PocketGPT/releases/download/"
    if (not re.fullmatch(r"[0-9a-f]{64}", manifest["sha256"]) or
            not manifest["url"].startswith(prefix) or
            type(manifest["bytes"]) is not int or manifest["bytes"] < 1):
        raise ValueError("Invalid PocketGPT release manifest")
    output = Path(output)

    def verify(path):
        if path.stat().st_size != manifest["bytes"] or checksum(path) != manifest["sha256"]:
            raise ValueError("Model checksum/size mismatch; existing files are not overwritten")

    if output.exists():
        verify(output)
        return {"model": str(output), "sha256": manifest["sha256"], "status": "verified"}
    if verify_only:
        raise FileNotFoundError("Model file does not exist: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pocketgpt-download-", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            request = Request(manifest["url"], headers={"User-Agent": "PocketGPT-model-downloader/1.0"})
            with urlopen(request, timeout=60) as response:
                downloaded = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    downloaded += len(block)
                    if downloaded > manifest["bytes"]:
                        raise ValueError("Download is larger than the release manifest")
                    stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        verify(temporary)
        # Publish only complete, verified bytes; never replace an existing file.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"model": str(output), "sha256": manifest["sha256"], "status": "downloaded_and_verified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="weights/pocketgpt-v1.pt")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download_model(args.output, args.verify_only), indent=2))
