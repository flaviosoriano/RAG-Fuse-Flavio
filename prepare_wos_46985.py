#!/usr/bin/env python3
"""Download, validate, and prepare WOS-46985 for RAG-Fuse."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import random
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
from sklearn.model_selection import StratifiedKFold


MENDELEY_PAGE_URL = "https://data.mendeley.com/datasets/9rw3vkcfy4/2"
MENDELEY_DOI = "10.17632/9rw3vkcfy4.2"
MENDELEY_LICENSE = "CC BY 4.0"
PUBLIC_DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/"
    "c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded"
)
HF_MIRROR_URL = "https://huggingface.co/datasets/HDLTex/web_of_science"
EXPECTED_NUM_SAMPLES = 46985
EXPECTED_NUM_LABELS = 134
EXPECTED_NUM_PARENT_LABELS = 7
EXPECTED_NUM_LEVEL2_CODES = 53
EXPECTED_ARCHIVE_FILES = {
    "Meta-data/Data.xlsx",
    "WOS46985/X.txt",
    "WOS46985/Y.txt",
    "WOS46985/YL1.txt",
    "WOS46985/YL2.txt",
}

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--download-url",
        default=PUBLIC_DOWNLOAD_URL,
        help="Archive download URL. Defaults to the validated public Mendeley file.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Optional pre-downloaded archive path. When omitted, the archive is downloaded.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse --archive-path instead of downloading a new archive.",
    )
    parser.add_argument(
        "--skip-hf-staging",
        action="store_true",
        help="Skip generation of the Hugging Face staging folder.",
    )
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(destination)


def col_ref_to_index(cell_ref: str) -> int:
    letters = ""
    for char in cell_ref:
        if char.isalpha():
            letters += char
        else:
            break
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def read_xlsx_records(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = []
        shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in shared_root.findall("a:si", XLSX_NS):
            value = "".join(text.text or "" for text in si.iterfind(".//a:t", XLSX_NS))
            shared_strings.append(value)

        worksheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = worksheet.find("a:sheetData", XLSX_NS).findall("a:row", XLSX_NS)

    records: list[dict[str, str]] = []
    header: list[str] | None = None
    for row in rows:
        values: dict[int, str] = {}
        for cell in row.findall("a:c", XLSX_NS):
            index = col_ref_to_index(cell.attrib["r"])
            value_node = cell.find("a:v", XLSX_NS)
            value = "" if value_node is None else value_node.text or ""
            if cell.attrib.get("t") == "s" and value:
                value = shared_strings[int(value)]
            values[index] = value

        if not values:
            continue

        width = max(values) + 1
        ordered_values = [values.get(i, "") for i in range(width)]

        if header is None:
            header = ordered_values
            continue

        if len(ordered_values) < len(header):
            ordered_values.extend([""] * (len(header) - len(ordered_values)))
        records.append(dict(zip(header, ordered_values[: len(header)])))

    if header is None:
        raise RuntimeError(f"No header row found in {path}")

    return records


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def load_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n").rstrip("\r") for line in handle]


def deterministic_train_val_split(
    ids: list[int], split_ratio: float = 0.9, seed: int = 72
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    boundary = int(split_ratio * len(shuffled))
    return shuffled[:boundary], shuffled[boundary:]


def build_label_cls(samples: list[dict]) -> dict[int, list[str]]:
    labels_counter: Counter[int] = Counter()
    for sample in samples:
        labels_counter.update(sample["labels_ids"])

    cutoff = round(0.2 * len(labels_counter))
    label_cls: dict[int, list[str]] = {}
    head: list[int] = []
    tail: list[int] = []

    for label_idx, _ in labels_counter.most_common():
        if len(head) <= cutoff:
            label_cls[label_idx] = ["all", "head"]
            head.append(label_idx)
        else:
            label_cls[label_idx] = ["all", "tail"]
            tail.append(label_idx)

    return label_cls


def get_text_cls(label_ids: Iterable[int], label_cls: dict[int, list[str]]) -> list[str]:
    classes = {"all"}
    for label_id in label_ids:
        classes.update(label_cls[label_id][1:])
    return sorted(classes, key=lambda value: (value != "all", value))


def build_folds(samples: list[dict], seed: int = 72) -> list[dict]:
    ids = [sample["idx"] for sample in samples]
    cls = [sample["labels_ids"][0] for sample in samples]
    splitter = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    folds = []
    for fold_idx, (train_val_indices, test_indices) in enumerate(splitter.split(ids, cls)):
        train_val_ids = [ids[i] for i in train_val_indices.tolist()]
        train_ids, val_ids = deterministic_train_val_split(
            train_val_ids, split_ratio=0.9, seed=seed + fold_idx
        )
        test_ids = [ids[i] for i in test_indices.tolist()]
        folds.append(
            {
                "fold_idx": fold_idx,
                "splits": {
                    "train": train_ids,
                    "val": val_ids,
                    "test": test_ids,
                },
            }
        )
    return folds


def checkpoint_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def checkpoint_folds(folds: list[dict], dataset_dir: Path) -> None:
    for fold in folds:
        fold_dir = dataset_dir / f"fold_{fold['fold_idx']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split, split_ids in fold["splits"].items():
            checkpoint_pickle(split_ids, fold_dir / f"{split}.pkl")


def format_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in counter.most_common()}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def copy_tree_contents(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    raw_dir = repo_root / "resource" / "dataset" / "raw_dataset" / "WOS-46985"
    processed_dir = repo_root / "resource" / "dataset" / "WOS-46985"
    hf_staging_dir = repo_root / "resource" / "dataset" / "hf_staging" / "wos-46985-rag-fuse"

    archive_path = args.archive_path
    if archive_path is None:
        archive_path = raw_dir / "downloads" / "wos_official_bundle.zip"
    archive_path = archive_path.resolve()

    if not args.skip_download:
        download_file(args.download_url, archive_path)
    elif not archive_path.exists():
        raise FileNotFoundError(
            f"--skip-download was provided, but archive was not found: {archive_path}"
        )

    archive_sha256 = sha256sum(archive_path)

    with tempfile.TemporaryDirectory(prefix="wos46985_") as temp_dir:
        extracted_dir = Path(temp_dir) / "extracted"
        extract_archive(archive_path, extracted_dir)

        archive_names = set()
        with zipfile.ZipFile(archive_path) as zf:
            archive_names = set(zf.namelist())

        missing_archive_files = EXPECTED_ARCHIVE_FILES - archive_names
        if missing_archive_files:
            raise RuntimeError(
                f"Archive is missing required files: {sorted(missing_archive_files)}"
            )

        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "Meta-data").mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted_dir / "WOS46985" / "X.txt", raw_dir / "X.txt")
        shutil.copy2(extracted_dir / "WOS46985" / "Y.txt", raw_dir / "Y.txt")
        shutil.copy2(extracted_dir / "WOS46985" / "YL1.txt", raw_dir / "YL1.txt")
        shutil.copy2(extracted_dir / "WOS46985" / "YL2.txt", raw_dir / "YL2.txt")
        shutil.copy2(extracted_dir / "Meta-data" / "Data.xlsx", raw_dir / "Meta-data" / "Data.xlsx")

    texts = load_lines(raw_dir / "X.txt")
    y_values = [int(line.strip()) for line in load_lines(raw_dir / "Y.txt")]
    yl1_values = [int(line.strip()) for line in load_lines(raw_dir / "YL1.txt")]
    yl2_values = [int(line.strip()) for line in load_lines(raw_dir / "YL2.txt")]
    metadata_records = read_xlsx_records(raw_dir / "Meta-data" / "Data.xlsx")

    if not (len(texts) == len(y_values) == len(yl1_values) == len(yl2_values) == len(metadata_records)):
        raise RuntimeError(
            "WOS-46985 raw file lengths do not agree: "
            f"X={len(texts)}, Y={len(y_values)}, YL1={len(yl1_values)}, "
            f"YL2={len(yl2_values)}, metadata={len(metadata_records)}"
        )

    if len(texts) != EXPECTED_NUM_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_SAMPLES} samples, found {len(texts)}"
        )

    unique_y = len(set(y_values))
    unique_yl1 = len(set(yl1_values))
    unique_yl2 = len(set(yl2_values))
    if unique_y != EXPECTED_NUM_LABELS:
        raise RuntimeError(f"Expected {EXPECTED_NUM_LABELS} Y labels, found {unique_y}")
    if unique_yl1 != EXPECTED_NUM_PARENT_LABELS:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_PARENT_LABELS} YL1 labels, found {unique_yl1}"
        )
    if unique_yl2 != EXPECTED_NUM_LEVEL2_CODES:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_LEVEL2_CODES} YL2 labels, found {unique_yl2}"
        )

    by_y_area: dict[int, Counter[str]] = defaultdict(Counter)
    by_y_domain: dict[int, Counter[str]] = defaultdict(Counter)
    metadata_alignment_mismatches = 0
    for idx, record in enumerate(metadata_records):
        area = normalize_space(record["area"])
        domain = normalize_space(record["Domain"])
        abstract = normalize_space(record["Abstract"])
        by_y_area[int(record["Y"].strip())][area] += 1
        by_y_domain[int(record["Y"].strip())][domain] += 1
        if abstract != normalize_space(texts[idx]):
            metadata_alignment_mismatches += 1

    label_text_map = {
        label_id: area_counter.most_common(1)[0][0]
        for label_id, area_counter in by_y_area.items()
    }
    label_domain_map = {
        label_id: domain_counter.most_common(1)[0][0]
        for label_id, domain_counter in by_y_domain.items()
    }
    ambiguous_label_names = {
        str(label_id): format_counter(area_counter)
        for label_id, area_counter in by_y_area.items()
        if len(area_counter) > 1
    }

    if len(label_text_map) != EXPECTED_NUM_LABELS:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_LABELS} label names, found {len(label_text_map)}"
        )
    if len(set(label_text_map.values())) != EXPECTED_NUM_LABELS:
        raise RuntimeError("Normalized label names are not globally unique")

    samples = []
    text_to_idx: dict[str, int] = {}
    for idx, text in enumerate(texts):
        text_idx = text_to_idx.setdefault(text, len(text_to_idx))
        label_id = y_values[idx]
        samples.append(
            {
                "idx": idx,
                "text_idx": text_idx,
                "text": text,
                "labels_ids": [label_id],
                "labels": [label_text_map[label_id]],
            }
        )

    label_cls = build_label_cls(samples)
    text_cls = {
        sample["text_idx"]: get_text_cls(sample["labels_ids"], label_cls) for sample in samples
    }
    folds = build_folds(samples, seed=72)
    relevance_map = {sample["text_idx"]: sample["labels_ids"] for sample in samples}
    labels_map = dict(sorted(label_text_map.items()))

    processed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_pickle(samples, processed_dir / "samples.pkl")
    checkpoint_folds(folds, processed_dir)
    checkpoint_pickle(relevance_map, processed_dir / "relevance_map.pkl")
    checkpoint_pickle(label_cls, processed_dir / "label_cls.pkl")
    checkpoint_pickle(text_cls, processed_dir / "text_cls.pkl")
    checkpoint_pickle(labels_map, processed_dir / "labels_map.pkl")

    raw_manifest = []
    for path in [
        raw_dir / "X.txt",
        raw_dir / "Y.txt",
        raw_dir / "YL1.txt",
        raw_dir / "YL2.txt",
        raw_dir / "Meta-data" / "Data.xlsx",
        archive_path,
    ]:
        raw_manifest.append(
            {
                "path": display_path(path, repo_root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256sum(path),
            }
        )

    metadata = {
        "dataset_name": "WOS-46985",
        "variant": "WOS46985",
        "n_samples": len(samples),
        "n_texts": len(text_to_idx),
        "n_leaf_labels": unique_y,
        "n_parent_labels": unique_yl1,
        "n_level2_codes": unique_yl2,
        "n_parent_child_pairs": len(set(zip(yl1_values, yl2_values))),
        "n_label_triplets": len(set(zip(y_values, yl1_values, yl2_values))),
        "label_semantics": {
            "Y": "flat target label",
            "YL1": "parent label",
            "YL2": "level-2 code reused across parents; not a globally unique flat label by itself",
        },
        "source": {
            "canonical_page_url": MENDELEY_PAGE_URL,
            "doi": MENDELEY_DOI,
            "license": MENDELEY_LICENSE,
            "download_url": args.download_url,
            "hf_mirror_url": HF_MIRROR_URL,
            "downloaded_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "archive_sha256": archive_sha256,
        },
        "source_files": {
            "X.txt": display_path(raw_dir / "X.txt", repo_root),
            "Y.txt": display_path(raw_dir / "Y.txt", repo_root),
            "YL1.txt": display_path(raw_dir / "YL1.txt", repo_root),
            "YL2.txt": display_path(raw_dir / "YL2.txt", repo_root),
            "Data.xlsx": display_path(raw_dir / "Meta-data" / "Data.xlsx", repo_root),
        },
        "label_domain_map": {str(k): v for k, v in sorted(label_domain_map.items())},
        "label_text_map": {str(k): v for k, v in sorted(label_text_map.items())},
        "ambiguous_label_text_candidates": ambiguous_label_names,
        "metadata_alignment_mismatches": metadata_alignment_mismatches,
        "raw_manifest": raw_manifest,
    }
    write_text(processed_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")

    source_metadata = {
        "dataset_name": "WOS-46985",
        "canonical_page_url": MENDELEY_PAGE_URL,
        "doi": MENDELEY_DOI,
        "license": MENDELEY_LICENSE,
        "download_url": args.download_url,
        "hf_mirror_url": HF_MIRROR_URL,
        "archive_sha256": archive_sha256,
        "downloaded_at_utc": metadata["source"]["downloaded_at_utc"],
    }
    write_text(raw_dir / "SOURCE.json", json.dumps(source_metadata, indent=2) + "\n")

    processed_readme = """# WOS-46985 for RAG-Fuse

This directory contains the processed WOS-46985 artifacts required by the RAG-Fuse pipeline.

- Source page: https://data.mendeley.com/datasets/9rw3vkcfy4/2
- DOI: 10.17632/9rw3vkcfy4.2
- License: CC BY 4.0
- Mirror used only for verification: https://huggingface.co/datasets/HDLTex/web_of_science

Processing notes:

- The flat target for RAG-Fuse is `Y` with 134 labels.
- `YL1` contains the 7 parent domains.
- `YL2` is not globally unique across parents and must not be used as the flat label id.
- `labels_map.pkl` stores `label_id -> label_text` derived from the official `Meta-data/Data.xlsx`.
- Eight `Y` ids have multiple raw `area` strings in the metadata file. The processed dataset uses the majority `area` per `Y` and records the alternatives in `metadata.json`.
"""
    write_text(processed_dir / "README.md", processed_readme)

    if not args.skip_hf_staging:
        raw_stage = hf_staging_dir / "raw" / "original"
        processed_stage = hf_staging_dir / "processed" / "rag_fuse"
        copy_tree_contents(raw_dir, raw_stage)
        copy_tree_contents(processed_dir, processed_stage)

        hf_readme = f"""---
license: cc-by-4.0
pretty_name: WOS-46985 for RAG-Fuse
language:
- en
tags:
- text-classification
- label-retrieval
- scientific-publications
- rag-fuse
size_categories:
- 10K<n<100K
---

# WOS-46985 for RAG-Fuse

This repository packages the official WOS-46985 raw files and the processed artifacts required by the RAG-Fuse pipeline.

## Provenance

- Canonical dataset page: {MENDELEY_PAGE_URL}
- DOI: `{MENDELEY_DOI}`
- License: `{MENDELEY_LICENSE}`
- Verification mirror only: {HF_MIRROR_URL}

## Layout

- `raw/original/`: official `X.txt`, `Y.txt`, `YL1.txt`, `YL2.txt`, plus `Meta-data/Data.xlsx`
- `processed/rag_fuse/`: `samples.pkl`, `relevance_map.pkl`, `label_cls.pkl`, `text_cls.pkl`, `labels_map.pkl`, `fold_0..9`, and `metadata.json`

## RAG-Fuse interpretation

- `Y` is the canonical 134-class flat label.
- `YL1` is the 7-class parent label.
- `YL2` is reused across parents and is not a globally unique flat label by itself.
- `labels_map.pkl` stores `label_id -> label_text`, where `label_text` is derived from the official metadata sheet.

## Notes on label names

The official metadata file contains eight `Y` ids with more than one raw `area` string. The processed dataset resolves each of those ids to the majority `area` string and records the alternatives in `processed/rag_fuse/metadata.json`.
"""
        write_text(hf_staging_dir / "README.md", hf_readme)

    print(f"Prepared raw dataset in {raw_dir}")
    print(f"Prepared processed dataset in {processed_dir}")
    if not args.skip_hf_staging:
        print(f"Prepared Hugging Face staging folder in {hf_staging_dir}")
    print(f"Unique Y labels: {unique_y}")
    print(f"Unique YL1 labels: {unique_yl1}")
    print(f"Unique YL2 labels: {unique_yl2}")
    print(f"Ambiguous Y -> area mappings resolved by majority vote: {len(ambiguous_label_names)}")


if __name__ == "__main__":
    main()
