"""Explicitly rebuild the final package file inventory after reviewed edits."""
import argparse
import json
from pathlib import Path
from .io import sha
from .verify import package_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = {"schema": "wikiflow.final.manifest.v1", "files_sha256":
                {name: sha(root/name) for name in package_files(root)}}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Manifest files:",len(manifest["files_sha256"]))


if __name__ == "__main__":
    main()
