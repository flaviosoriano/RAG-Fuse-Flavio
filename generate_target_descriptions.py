"""
Generate target_descriptions.pkl for a given dataset using OpenAI Batch API.

Overview:
  1. Loads `samples.pkl` from `resource/dataset/{DATASET}/`
  2. Builds a training-only candidate pool from the union of all
     `fold_*/train.pkl` sample ids
  3. Extracts labels and redistributes a fixed total sample budget across them
     using `--sampling_strategy`
  4. Builds one prompt per label from the selected training samples
  5. Either:
     - prints a dry-run summary with token / cost estimates, or
     - generates descriptions via the OpenAI API in one or more sequential batches
  6. Saves `{label: description}` to
     `resource/llm/{DATASET}/target_descriptions.pkl`

Notes:
  - `--max_samples` is a global sampling budget across all labels
  - `--sampling_strategy` controls how that budget is redistributed:
      `uniform`, `inverse_freq`, `sqrt_inverse_freq`
  - Dry-run mode never writes `target_descriptions.pkl`
  - The head / tail dry-run summary uses the built-in `label_cls.pkl` artifact

Arguments:
  - `--dataset <name>`
      Dataset folder under `resource/dataset/`.
      Examples in this repo: `ACM`, `OHSUMED`, `RCV1`, `REUTERS`, `TWITTER`.

  - `--max_samples <int>`
      Total sampling budget across all labels.
      Default: `256`.

  - `--sampling_strategy <value>`
      Accepted values:
        `uniform`
          Split the total budget as evenly as possible across labels.
        `inverse_freq`
          Allocate more of the budget to rarer labels using inverse label frequency.
        `sqrt_inverse_freq`
          Similar to `inverse_freq`, but with a softer rare-label boost.
      Default: `uniform`.

  - `--max_text_tokens <int>`
      Maximum number of words taken from each sampled text when building the prompt.
      Default: `128`.

  - `--model <name>`
      OpenAI chat model used for generation and token/cost estimation.
      Examples supported by the pricing table include:
      `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`,
      `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o3-mini`.
      Default: `gpt-4o`.

  - `--temperature <float>`
      Sampling temperature for generation.
      Default: `0.7`.

  - `--seed <int>`
      Random seed for reproducible sample selection.
      Default: `42`.

  - `--dry_run`
      Build prompts and print selection/token/cost summaries only.
      No API call is made and no `.pkl` file is written.

  - `--output <path>`
      Optional override for the final pickle path.
      Default: `resource/llm/{DATASET}/target_descriptions.pkl`.

  - `--sync`
      Use synchronous API calls instead of the Batch API.

  - `--num_batches <int>`
      Split the generated label-description requests into this many
      sequential batches. Requests are partitioned as evenly as possible and
      each batch is fully completed and persisted to the `.pkl` accumulator
      before the next batch starts.
      Default: `1`.

  - `--poll_interval <int>`
      Seconds between batch status checks.
      Default: `30`.

  - `--batch_id <id>`
      Resume polling an existing batch job using the saved batch mapping file.

Usage:
    export OPENAI_API_KEY="sk-..."

    # Default batch mode: submit one request per label and wait for completion
    python generate_target_descriptions.py --dataset DBLP
    python generate_target_descriptions.py --dataset DBLP --model gpt-4o
    python generate_target_descriptions.py --dataset DBLP --num_batches 4

    # Fixed-budget sampling with rare-label emphasis
    python generate_target_descriptions.py --dataset DBLP --max_samples 64
    python generate_target_descriptions.py --dataset DBLP --max_samples 64 --sampling_strategy inverse_freq
    python generate_target_descriptions.py --dataset DBLP --max_samples 64 --sampling_strategy sqrt_inverse_freq

    # Resume an already-submitted batch without rebuilding or reuploading prompts
    python generate_target_descriptions.py --dataset DBLP --batch_id batch_abc123

    # Synchronous mode: generate immediately instead of using the Batch API
    python generate_target_descriptions.py --dataset DBLP --sync

    # Dry run: build prompts, show per-label selection + head/tail summaries,
    # and estimate tokens / cost without calling the API or writing a .pkl
    python generate_target_descriptions.py --dataset DBLP --model gpt-4o --dry_run
    python generate_target_descriptions.py --dataset DBLP --max_samples 48 --dry_run
"""

import argparse
import json
import logging
import math
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing per 1M tokens (USD) – update as needed
# ---------------------------------------------------------------------------
MODEL_PRICING = {
    # model_name: (input_cost_per_1M, output_cost_per_1M)
    "gpt-4":        (30.00, 60.00),
    "gpt-4-turbo":  (10.00, 30.00),
    "gpt-4o":       ( 2.50,  10.00),
    "gpt-4o-mini":  ( 0.15,   0.60),
    "gpt-4.1":      ( 2.00,   8.00),
    "gpt-4.1-mini": ( 0.40,   1.60),
    "gpt-4.1-nano": ( 0.10,   0.40),
    "o3-mini":      ( 1.10,   4.40),
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "resource" / "dataset"
LLM_DIR = BASE_DIR / "resource" / "llm"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in `text` using tiktoken.
    Falls back to a word-based approximation (1 token ≈ 0.75 words) if
    tiktoken is not installed.
    """
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        # rough approximation
        return int(len(text.split()) / 0.75)


def count_chat_tokens(system_msg: str, user_msg: str, model: str = "gpt-4o") -> int:
    """
    Estimate total input tokens for a ChatCompletion request with
    one system message and one user message.
    Adds ~7 tokens of overhead per message for chat formatting.
    """
    overhead = 7  # per-message overhead (role, name, delimiters)
    return (
        count_tokens(system_msg, model)
        + count_tokens(user_msg, model)
        + overhead * 2
        + 3  # reply priming
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_samples(dataset: str) -> list[dict]:
    """Load samples.pkl for the given dataset."""
    path = DATASET_DIR / dataset / "samples.pkl"
    if not path.exists():
        raise FileNotFoundError(f"samples.pkl not found at {path}")
    with open(path, "rb") as f:
        samples = pickle.load(f)
    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples


def load_aggregated_train_ids(dataset: str) -> set[int]:
    """Load and aggregate all train.pkl sample ids across folds."""
    dataset_path = DATASET_DIR / dataset
    fold_dirs = sorted(path for path in dataset_path.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {dataset_path}")

    aggregated_ids: set[int] = set()
    loaded_folds = 0

    for fold_dir in fold_dirs:
        train_path = fold_dir / "train.pkl"
        if not train_path.exists():
            logger.warning(f"Skipping missing train split: {train_path}")
            continue

        with open(train_path, "rb") as f:
            split_ids = pickle.load(f)
        aggregated_ids.update(split_ids)
        loaded_folds += 1
        logger.info(f"Loaded {len(split_ids)} train ids from {train_path}")

    if not aggregated_ids:
        raise FileNotFoundError(
            f"No train.pkl ids could be loaded for dataset '{dataset}' under {dataset_path}"
        )

    logger.info(
        f"Aggregated {len(aggregated_ids)} unique train sample ids across {loaded_folds} fold(s)"
    )
    return aggregated_ids


def filter_samples_by_ids(samples: list[dict], allowed_ids: set[int]) -> list[dict]:
    """Filter samples by sample idx."""
    filtered = [sample for sample in samples if sample["idx"] in allowed_ids]
    logger.info(
        f"Filtered samples to {len(filtered)} training-only samples "
        f"(from {len(samples)} total)"
    )
    return filtered


def extract_labels(samples: list[dict]) -> list[str]:
    """Return sorted list of unique labels across all samples."""
    labels = set()
    for sample in samples:
        for label in sample["labels"]:
            labels.add(label)
    return sorted(labels)


def extract_label_to_idx(samples: list[dict]) -> dict[str, int]:
    """Return a stable label -> label_idx mapping from samples."""
    label_to_idx: dict[str, int] = {}
    for sample in samples:
        labels = sample.get("labels", [])
        labels_ids = sample.get("labels_ids", [])
        for label, label_idx in zip(labels, labels_ids):
            label_to_idx[label] = label_idx
    return label_to_idx


def compute_label_frequencies(samples: list[dict]) -> dict[str, int]:
    """Count positive sample frequency per label."""
    frequencies: dict[str, int] = defaultdict(int)
    for sample in samples:
        for label in sample["labels"]:
            frequencies[label] += 1
    return dict(frequencies)


def load_label_buckets(
    dataset: str,
    label_to_idx: dict[str, int],
) -> dict[str, str]:
    """Load built-in head/tail bucket assignments from label_cls.pkl."""
    path = DATASET_DIR / dataset / "label_cls.pkl"
    if not path.exists():
        logger.warning(f"label_cls.pkl not found at {path}; head/tail summary unavailable")
        return {}

    with open(path, "rb") as f:
        label_cls = pickle.load(f)

    label_to_bucket: dict[str, str] = {}
    for label, label_idx in label_to_idx.items():
        classes = label_cls.get(label_idx, [])
        if "head" in classes:
            label_to_bucket[label] = "head"
        elif "tail" in classes:
            label_to_bucket[label] = "tail"

    return label_to_bucket


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def get_requested_samples_per_label(
    labels: list[str],
    label_frequencies: dict[str, int],
    total_budget: int,
    strategy: str,
) -> dict[str, int]:
    """
    Compute the target number of samples per label.

    `total_budget` is redistributed across labels.
    `inverse_freq` and `sqrt_inverse_freq` shift more of that shared budget
    toward rarer labels.
    """
    if total_budget < 1:
        raise ValueError("total_budget must be at least 1")

    if strategy == "uniform":
        base_count, remainder = divmod(total_budget, len(labels))
        return {
            label: base_count + (1 if idx < remainder else 0)
            for idx, label in enumerate(labels)
        }

    weights: dict[str, float] = {}
    for label in labels:
        freq = label_frequencies[label]
        if strategy == "inverse_freq":
            weights[label] = 1.0 / freq
        elif strategy == "sqrt_inverse_freq":
            weights[label] = 1.0 / math.sqrt(freq)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")

    weight_sum = sum(weights.values())
    requested: dict[str, int] = {}
    fractional_parts: list[tuple[float, str]] = []
    allocated_total = 0

    for label in labels:
        raw = total_budget * (weights[label] / weight_sum)
        whole = math.floor(raw)
        requested[label] = whole
        allocated_total += whole
        fractional_parts.append((raw - whole, label))

    remaining = total_budget - allocated_total
    for _, label in sorted(fractional_parts, reverse=True):
        if remaining <= 0:
            break
        requested[label] += 1
        remaining -= 1

    return requested


def select_samples(
    samples: list[dict],
    labels: list[str],
    label_frequencies: dict[str, int],
    total_budget: int,
    sampling_strategy: str = "uniform",
    seed: int = 42,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int | float | str]]]:
    """
    Select samples for each label from the training-only pool.
    Returns:
      - {label: [sample dicts]}
      - {label: selection metadata}
    """
    rng = random.Random(seed)

    label_to_samples: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        for label in sample["labels"]:
            label_to_samples[label].append(sample)

    requested_counts = get_requested_samples_per_label(
        labels=labels,
        label_frequencies=label_frequencies,
        total_budget=total_budget,
        strategy=sampling_strategy,
    )

    selected: dict[str, list[dict]] = {}
    selection_stats: dict[str, dict[str, int | float | str]] = {}
    for label in labels:
        pool = label_to_samples[label]
        requested = requested_counts[label]
        k = min(requested, len(pool))
        selected[label] = rng.sample(pool, k)
        selection_stats[label] = {
            "requested": requested,
            "selected": k,
            "available": len(pool),
            "frequency": label_frequencies[label],
            "strategy": sampling_strategy,
        }
        logger.info(
            f"  Label '{label}': selected {k}/{len(pool)} samples "
            f"(requested={requested}, freq={label_frequencies[label]})"
        )

    total = sum(len(v) for v in selected.values())
    logger.info(
        f"Total selected samples: {total} "
        f"(requested budget={total_budget}, strategy={sampling_strategy})"
    )
    return selected, selection_stats


def summarize_bucket_stats(
    labels: list[str],
    per_label_tokens: list[dict],
    selection_stats: dict[str, dict[str, int | float | str]],
    label_to_bucket: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Aggregate dry-run stats by built-in head/tail buckets."""
    bucket_stats: dict[str, dict[str, int]] = {
        "head": {"labels": 0, "selected_samples": 0, "input_tokens": 0, "output_tokens_est": 0},
        "tail": {"labels": 0, "selected_samples": 0, "input_tokens": 0, "output_tokens_est": 0},
    }

    tokens_by_label = {entry["label"]: entry for entry in per_label_tokens}
    for label in labels:
        bucket = label_to_bucket.get(label)
        if bucket not in bucket_stats:
            continue

        bucket_stats[bucket]["labels"] += 1
        bucket_stats[bucket]["selected_samples"] += int(selection_stats[label]["selected"])
        bucket_stats[bucket]["input_tokens"] += int(tokens_by_label[label]["input_tokens"])
        bucket_stats[bucket]["output_tokens_est"] += int(tokens_by_label[label]["output_tokens_est"])

    return bucket_stats


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_context_for_label(
    label: str,
    label_samples: list[dict],
    max_text_tokens: int = 128,
) -> str:
    """
    Build the text context block for a label.
    Each sample contributes truncated text + its labels.
    """
    lines = []
    for sample in label_samples:
        truncated_text = " ".join(sample["text"].split()[:max_text_tokens])
        sample_labels = "; ".join(sample["labels"])
        lines.append(f"  Text: {truncated_text}")
        lines.append(f"  Labels: {sample_labels}")
        lines.append("")
    return "\n".join(lines)


def build_gpt4_prompt(label: str, context_block: str) -> str:
    """
    Build the prompt sent to GPT-4 to generate a target description
    for a given label based on its associated texts.
    """
    return (
        f"You are an expert in text classification and domain analysis.\n\n"
        f"Below is a set of texts and their associated labels from a dataset. "
        f"Your task is to generate a concise and informative description for the label '{label}'.\n\n"
        f"The description should:\n"
        f"- Capture the essence of the label as reflected in the texts\n"
        f"- Focus on technical aspects, key concepts, techniques, and challenges\n"
        f"- Be a single paragraph (3-5 sentences)\n\n"
        f"Texts and Labels:\n"
        f"{context_block}\n"
        f"Now, provide a detailed and accurate description for the label '{label}':\n"
    )


# ---------------------------------------------------------------------------
# GPT-4 interaction
# ---------------------------------------------------------------------------
def call_gpt4(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_retries: int = 5,
) -> str:
    """Call OpenAI ChatCompletion API with retry logic."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it before running: export OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=api_key)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a domain expert that generates precise, "
                            "technical descriptions of text classification labels."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2**attempt
            logger.warning(
                f"GPT-4 call failed (attempt {attempt}/{max_retries}): {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"GPT-4 call failed after {max_retries} retries")


# ---------------------------------------------------------------------------
# OpenAI client helper
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are a domain expert that generates precise, "
    "technical descriptions of text classification labels."
)


def get_openai_client():
    """Return an authenticated OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it before running: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Batch API functions
# ---------------------------------------------------------------------------
def prepare_batch_jsonl(
    labels: list[str],
    prompts: dict[str, str],
    model: str,
    temperature: float,
    max_tokens: int = 512,
    output_dir: Path = None,
    artifact_suffix: str | None = None,
) -> tuple[str, dict[str, str]]:
    """
    Write a .jsonl batch input file and a mapping JSON.
    Returns (jsonl_path, {custom_id: label} mapping).
    """
    id_to_label = {}
    if artifact_suffix:
        jsonl_name = f"batch_input_{artifact_suffix}.jsonl"
        mapping_name = f"batch_mapping_{artifact_suffix}.json"
    else:
        jsonl_name = "batch_input.jsonl"
        mapping_name = "batch_mapping.json"

    jsonl_path = (output_dir / jsonl_name) if output_dir else Path(jsonl_name)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w") as f:
        for i, label in enumerate(labels):
            custom_id = f"label-{i}"
            id_to_label[custom_id] = label
            line = json.dumps({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompts[label]},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            })
            f.write(line + "\n")

    # Save mapping so we can resume with --batch_id later
    mapping_path = jsonl_path.parent / mapping_name
    with open(mapping_path, "w") as f:
        json.dump(id_to_label, f, indent=2)

    logger.info(f"Batch input written: {jsonl_path} ({len(labels)} requests)")
    logger.info(f"Label mapping saved: {mapping_path}")
    return str(jsonl_path), id_to_label


def partition_labels_evenly(labels: list[str], num_batches: int) -> list[list[str]]:
    """Split labels into `num_batches` ordered partitions as evenly as possible."""
    base_size, remainder = divmod(len(labels), num_batches)
    partitions: list[list[str]] = []
    start = 0

    for batch_idx in range(num_batches):
        batch_size = base_size + (1 if batch_idx < remainder else 0)
        end = start + batch_size
        partitions.append(labels[start:end])
        start = end

    return partitions


def load_description_accumulator(output_path: Path) -> dict[str, str]:
    """Load an existing target_descriptions.pkl accumulator if present."""
    if not output_path.exists():
        return {}

    with open(output_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Accumulator at {output_path} must contain a dict")

    return data


def persist_description_batch(
    output_path: Path,
    batch_descriptions: dict[str, str],
) -> dict[str, str]:
    """Merge one completed batch into the pickle accumulator and persist it."""
    accumulated = load_description_accumulator(output_path)
    accumulated.update(batch_descriptions)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(accumulated, f)

    logger.info(
        f"Persisted {len(batch_descriptions)} batch result(s) to {output_path} "
        f"({len(accumulated)} total)"
    )
    return accumulated


def submit_batch(client, jsonl_path: str):
    """Upload JSONL file and create a batch job. Returns the Batch object."""
    logger.info("Uploading batch input file to OpenAI...")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info(f"File uploaded: id={file_obj.id}")

    logger.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logger.info(f"Batch created: id={batch.id} | status={batch.status}")
    return batch


def poll_batch(client, batch_id: str, poll_interval: int = 30):
    """Poll batch status until a terminal state is reached."""
    terminal_states = {"completed", "failed", "expired", "cancelled"}
    logger.info(f"Polling batch {batch_id} every {poll_interval}s...")

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        logger.info(
            f"  status={batch.status} | "
            f"completed={counts.completed}/{counts.total} | "
            f"failed={counts.failed}"
        )
        if batch.status in terminal_states:
            return batch
        time.sleep(poll_interval)


def download_batch_results(client, output_file_id: str) -> list[dict]:
    """Download and parse the batch output JSONL."""
    logger.info(f"Downloading results (file: {output_file_id})...")
    content = client.files.content(output_file_id)
    text = content.text
    results = []
    for line in text.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    logger.info(f"Downloaded {len(results)} result(s)")
    return results


def parse_batch_results(
    results: list[dict],
    id_to_label: dict[str, str],
) -> dict[str, str]:
    """Parse batch output into {label: description} dict."""
    target_descriptions = {}
    errors = []
    for result in results:
        custom_id = result["custom_id"]
        label = id_to_label.get(custom_id, custom_id)

        if result.get("error"):
            errors.append((label, result["error"]))
            logger.error(f"  ERROR [{label}]: {result['error']}")
            continue

        response = result.get("response", {})
        if response.get("status_code") == 200:
            body = response["body"]
            content = body["choices"][0]["message"]["content"].strip()
            target_descriptions[label] = content
            logger.info(f"  [{label}]: {content[:120]}...")
        else:
            errors.append((label, response))
            logger.error(
                f"  NON-200 [{label}]: status_code={response.get('status_code')}"
            )

    if errors:
        logger.warning(f"{len(errors)} request(s) failed — see errors above.")

    return target_descriptions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate target_descriptions.pkl for a dataset using GPT-4o."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., DBLP, ACM, REUTERS, OHSUMED, TWITTER)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=256,
        help=(
            "Total sampling budget across labels, redistributed according to "
            "--sampling_strategy (default: 256)"
        ),
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "inverse_freq", "sqrt_inverse_freq"],
        help=(
            "How aggressively to favor rarer labels when deciding the per-label "
            "sample count. Default: uniform"
        ),
    )
    parser.add_argument(
        "--max_text_tokens",
        type=int,
        default=128,
        help="Max words per sample text to include in the context (default: 128)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o). Can also use gpt-4-turbo, etc.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for GPT-4o (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection (default: 42)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, print prompts without calling GPT-4o (useful for debugging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path override. Default: resource/llm/{dataset}/target_descriptions.pkl",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous API calls instead of Batch API (no 50%% discount)",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Seconds between batch status checks (default: 30)",
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default=None,
        help="Resume polling on an existing batch ID (skips prompt building & upload)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help=(
            "Split generated requests into this many sequential batches "
            "(default: 1)"
        ),
    )

    args = parser.parse_args()

    if args.num_batches < 1:
        parser.error("--num_batches must be at least 1")

    # ---- 1. Load data ----
    logger.info(f"=== Generating target descriptions for dataset: {args.dataset} ===")
    all_samples = load_samples(args.dataset)
    aggregated_train_ids = load_aggregated_train_ids(args.dataset)
    samples = filter_samples_by_ids(all_samples, aggregated_train_ids)
    labels = extract_labels(samples)
    label_frequencies = compute_label_frequencies(samples)
    label_to_idx = extract_label_to_idx(all_samples)
    label_to_bucket = load_label_buckets(args.dataset, label_to_idx)
    logger.info(f"Found {len(labels)} unique labels: {labels}")
    logger.info("Prompts will be built from the union of all fold train.pkl samples only")

    total_sampling_budget = args.max_samples

    # ---- 2. Select training samples per label ----
    logger.info(
        f"Selecting samples using strategy={args.sampling_strategy} "
        f"with total budget={total_sampling_budget}..."
    )
    selected, selection_stats = select_samples(
        samples=samples,
        labels=labels,
        label_frequencies=label_frequencies,
        total_budget=total_sampling_budget,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
    )
    label_batches = partition_labels_evenly(labels, args.num_batches)
    label_to_batch_idx = {
        label: batch_idx
        for batch_idx, batch_labels in enumerate(label_batches, start=1)
        for label in batch_labels
    }

    # ---- 3. Build prompts for all labels ----
    system_msg = SYSTEM_MSG
    max_output_tokens = 512

    # accumulators for dry-run token report
    total_input_tokens = 0
    total_output_tokens_est = 0
    per_label_tokens: list[dict] = []

    if args.dry_run and not HAS_TIKTOKEN:
        logger.warning(
            "tiktoken not installed – token counts are approximate. "
            "Install with: pip install tiktoken"
        )

    prompts: dict[str, str] = {}
    for label in labels:
        label_samples = selected[label]
        context_block = build_context_for_label(
            label, label_samples, max_text_tokens=args.max_text_tokens
        )
        prompts[label] = build_gpt4_prompt(label, context_block)

        if args.dry_run:
            input_tokens = count_chat_tokens(system_msg, prompts[label], model=args.model)
            total_input_tokens += input_tokens
            total_output_tokens_est += max_output_tokens
            per_label_tokens.append({
                "label": label,
                "bucket": label_to_bucket.get(label, "unknown"),
                "input_tokens": input_tokens,
                "output_tokens_est": max_output_tokens,
            })
            logger.info(f"\n{'='*60}")
            logger.info(f"[DRY RUN] Label: {label}")
            logger.info(
                f"Batch: {label_to_batch_idx[label]}/{args.num_batches} | "
                f"Bucket: {label_to_bucket.get(label, 'unknown')} | "
                f"Selected: {selection_stats[label]['selected']}/{selection_stats[label]['available']} "
                f"(requested={selection_stats[label]['requested']}, freq={selection_stats[label]['frequency']})"
            )
            logger.info(f"Prompt ({len(prompts[label])} chars / ~{input_tokens} input tokens)")
            logger.info(f"Estimated output: up to {max_output_tokens} tokens")
            logger.info(f"{'='*60}")

    if args.dry_run:
        bucket_stats = summarize_bucket_stats(
            labels=labels,
            per_label_tokens=per_label_tokens,
            selection_stats=selection_stats,
            label_to_bucket=label_to_bucket,
        )

        logger.info(f"\n{'='*60}")
        logger.info("DRY-RUN SELECTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Configured sequential batches: {args.num_batches}")
        logger.info(
            f"{'Label':<35} {'Batch':<8} {'Bucket':<8} {'Selected':>10} {'Avail':>10} {'Input':>10} {'Output':>10}"
        )
        logger.info(
            f"{'-'*35} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}"
        )
        for entry in per_label_tokens:
            stats = selection_stats[entry["label"]]
            batch_label = f"{label_to_batch_idx[entry['label']]}/{args.num_batches}"
            logger.info(
                f"{entry['label']:<35} "
                f"{batch_label:<8} "
                f"{entry['bucket']:<8} "
                f"{int(stats['selected']):>10,} "
                f"{int(stats['available']):>10,} "
                f"{entry['input_tokens']:>10,} "
                f"{entry['output_tokens_est']:>10,}"
            )

        logger.info(f"\n{'='*60}")
        logger.info("HEAD/TAIL DISTRIBUTION")
        logger.info(f"{'='*60}")
        logger.info(
            f"{'Bucket':<8} {'Labels':>10} {'Selected':>12} {'Input':>12} {'Output':>12}"
        )
        logger.info(
            f"{'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12}"
        )
        for bucket in ["head", "tail"]:
            stats = bucket_stats[bucket]
            logger.info(
                f"{bucket:<8} "
                f"{stats['labels']:>10,} "
                f"{stats['selected_samples']:>12,} "
                f"{stats['input_tokens']:>12,} "
                f"{stats['output_tokens_est']:>12,}"
            )

        logger.info(f"\n{'='*60}")
        logger.info("TOKEN & COST ESTIMATION")
        logger.info(f"{'='*60}")
        logger.info(f"{'Label':<35} {'Input':>10} {'Output (est)':>14}")
        logger.info(f"{'-'*35} {'-'*10} {'-'*14}")
        for entry in per_label_tokens:
            logger.info(
                f"{entry['label']:<35} {entry['input_tokens']:>10,} {entry['output_tokens_est']:>14,}"
            )
        logger.info(f"{'-'*35} {'-'*10} {'-'*14}")
        logger.info(
            f"{'TOTAL':<35} {total_input_tokens:>10,} {total_output_tokens_est:>14,}"
        )

        in_price, out_price = MODEL_PRICING.get(
            args.model, MODEL_PRICING.get("gpt-4")
        )
        input_cost_sync = (total_input_tokens / 1_000_000) * in_price
        output_cost_sync = (total_output_tokens_est / 1_000_000) * out_price
        total_cost_sync = input_cost_sync + output_cost_sync

        batch_discount = 0.5
        input_cost_batch = input_cost_sync * batch_discount
        output_cost_batch = output_cost_sync * batch_discount
        total_cost_batch = total_cost_sync * batch_discount

        logger.info(f"\nModel: {args.model}")
        logger.info(f"Pricing: ${in_price:.2f} / 1M input, ${out_price:.2f} / 1M output")
        if args.model not in MODEL_PRICING:
            logger.warning(
                f"Model '{args.model}' not in pricing table – using gpt-4o pricing as fallback. "
                f"Update MODEL_PRICING dict for accurate estimates."
            )
        logger.info("")
        logger.info(f"  {'':30} {'Sync':>12} {'Batch (-50%)':>14}")
        logger.info(f"  {'─'*30} {'─'*12} {'─'*14}")
        logger.info(f"  {'Input cost:':<30} ${input_cost_sync:>10.4f} ${input_cost_batch:>12.4f}")
        logger.info(f"  {'Output cost (est):':<30} ${output_cost_sync:>10.4f} ${output_cost_batch:>12.4f}")
        logger.info(f"  {'─'*30} {'─'*12} {'─'*14}")
        logger.info(f"  {'TOTAL:':<30} ${total_cost_sync:>10.4f} ${total_cost_batch:>12.4f}")
        logger.info("")
        logger.info(
            f"Note: output tokens estimated at max ({max_output_tokens}/label). "
            f"Actual usage will likely be lower."
        )
        logger.info("Dry run completed without writing target_descriptions.pkl")
        logger.info(f"{'='*60}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = LLM_DIR / args.dataset / "target_descriptions.pkl"

    non_empty_batches = [batch for batch in label_batches if batch]

    if not non_empty_batches:
        logger.error("No labels available to process.")
        return

    logger.info(
        f"Prepared {len(labels)} label request(s) across {args.num_batches} batch(es); "
        f"executing {len(non_empty_batches)} non-empty batch(es) sequentially"
    )

    # ---- 4. Generate descriptions ----
    target_descriptions: dict[str, str] = {}

    if args.batch_id:
        # ---- Resume: poll an existing batch ----
        llm_dir = LLM_DIR / args.dataset
        mapping_path = llm_dir / "batch_mapping.json"
        if not mapping_path.exists():
            logger.error(
                f"Cannot resume: {mapping_path} not found. "
                f"Run without --batch_id first to create a new batch."
            )
            return
        with open(mapping_path) as f:
            id_to_label = json.load(f)

        client = get_openai_client()
        batch = poll_batch(client, args.batch_id, args.poll_interval)

        if batch.status == "completed":
            results = download_batch_results(client, batch.output_file_id)
            target_descriptions = parse_batch_results(results, id_to_label)
            if target_descriptions:
                target_descriptions = persist_description_batch(
                    output_path,
                    target_descriptions,
                )
        else:
            logger.error(f"Batch ended with status: {batch.status}")
            if batch.error_file_id:
                logger.error(f"Error file ID: {batch.error_file_id}")
            return

    elif args.sync:
        # ---- Synchronous mode (old behavior, full price) ----
        logger.info("Using synchronous API calls (no batch discount)...")
        for batch_idx, batch_labels in enumerate(non_empty_batches, start=1):
            logger.info(
                f"Starting sync batch {batch_idx}/{len(non_empty_batches)} "
                f"with {len(batch_labels)} label(s)"
            )
            batch_descriptions: dict[str, str] = {}
            for label in batch_labels:
                logger.info(f"Generating description for label: '{label}'...")
                description = call_gpt4(
                    prompts[label],
                    model=args.model,
                    temperature=args.temperature,
                )
                batch_descriptions[label] = description
                logger.info(f"  -> Generated ({len(description)} chars): {description[:150]}...")

            target_descriptions = persist_description_batch(output_path, batch_descriptions)

    else:
        # ---- Batch mode (default – 50% cost discount) ----
        logger.info("Using Batch API (50% cost discount, up to 24h turnaround)...")
        llm_dir = LLM_DIR / args.dataset
        client = get_openai_client()

        for batch_idx, batch_labels in enumerate(non_empty_batches, start=1):
            logger.info(
                f"Starting Batch API batch {batch_idx}/{len(non_empty_batches)} "
                f"with {len(batch_labels)} label(s)"
            )
            batch_prompts = {label: prompts[label] for label in batch_labels}
            artifact_suffix = None
            if args.num_batches > 1:
                artifact_suffix = f"{batch_idx}_of_{len(non_empty_batches)}"

            jsonl_path, id_to_label = prepare_batch_jsonl(
                batch_labels,
                batch_prompts,
                model=args.model,
                temperature=args.temperature,
                max_tokens=max_output_tokens,
                output_dir=llm_dir,
                artifact_suffix=artifact_suffix,
            )

            batch = submit_batch(client, jsonl_path)
            logger.info(
                f"\n  >>> Batch ID: {batch.id}\n"
                f"  >>> Submitted batch {batch_idx}/{len(non_empty_batches)}\n"
                f"  >>> To resume later: python {Path(__file__).name} "
                f"--dataset {args.dataset} --batch_id {batch.id}\n"
            )

            batch = poll_batch(client, batch.id, args.poll_interval)

            if batch.status == "completed":
                results = download_batch_results(client, batch.output_file_id)
                batch_descriptions = parse_batch_results(results, id_to_label)
                if batch_descriptions:
                    target_descriptions = persist_description_batch(
                        output_path,
                        batch_descriptions,
                    )
            else:
                logger.error(f"Batch ended with status: {batch.status}")
                if batch.error_file_id:
                    logger.error(f"Error file ID: {batch.error_file_id}")
                logger.info(
                    f"You can retry later with: python {Path(__file__).name} "
                    f"--dataset {args.dataset} --batch_id {batch.id}"
                )
                return

    if not target_descriptions:
        logger.error("No descriptions generated. Aborting save.")
        return

    # ---- 5. target_descriptions.pkl already persisted incrementally ----
    logger.info(f"\nSaved target_descriptions.pkl at: {output_path}")
    logger.info(f"Contents: {len(target_descriptions)} label descriptions")

    # ---- 6. Summary ----
    logger.info("\n=== Summary ===")
    for label, desc in target_descriptions.items():
        logger.info(f"  [{label}]: {desc[:100]}...")

    logger.info("\nDone! You can now run prompt optimization:")
    logger.info(
        f"  python main.py tasks=[prompt_opt] data={args.dataset} data.text_features_source=TXT"
    )


if __name__ == "__main__":
    main()
