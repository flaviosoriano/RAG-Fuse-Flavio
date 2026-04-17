#!/usr/bin/env python3
"""Upload the prepared WOS-46985 staging folder to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, HfFolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination dataset repo id, for example `username/wos-46985-rag-fuse`.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("resource/dataset/hf_staging/wos-46985-rag-fuse"),
        help="Prepared staging folder to upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the target dataset repo as private.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload WOS-46985 raw and processed artifacts for RAG-Fuse",
        help="Commit message used for the upload.",
    )
    return parser.parse_args()


def resolve_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or HfFolder.get_token()
    )


def main() -> None:
    args = parse_args()
    token = resolve_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Authenticate with `huggingface-cli login` "
            "or export `HF_TOKEN`, then rerun this script."
        )

    folder = args.folder.resolve()
    if not folder.exists():
        raise SystemExit(f"Staging folder not found: {folder}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo="",
        commit_message=args.commit_message,
    )
    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
