# Rodar RAG-Fuse em `LBD-UFMG/RCV1-103-H3`

Este roteiro usa o dataset `LBD-UFMG/RCV1-103-H3` como golden truth, em um nome local separado: `RCV1-103-H3`.

Arquivos estaticos ja preparados:
- `setting/data/RCV1-103-H3.yaml`
- `run/RCV1-103-H3.sh`
- `resource/llm/RCV1-103-H3/seed_prompt.txt`
- `resource/llm/RCV1-103-H3/meta_prompt.txt`

O dataset remoto tem somente folds `0..4`. Nao rode `0 9` para este dataset.

## 1. Preparar ambiente

```bash
cd /root/RAG-Fuse-Flavio

eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate RAG-Fuse
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Se ainda nao estiver autenticado no Hugging Face:

```bash
/root/miniconda3/envs/RAG-Fuse/bin/huggingface-cli login
```

## 2. Baixar dataset golden truth

```bash
/root/miniconda3/envs/RAG-Fuse/bin/hf download LBD-UFMG/RCV1-103-H3 \
  --repo-type dataset \
  --local-dir resource/dataset/RCV1-103-H3
```

## 3. Validar formato do dataset

```bash
python - <<'PY'
import pickle
from pathlib import Path

base = Path("resource/dataset/RCV1-103-H3")

required = ["samples.pkl", "relevance_map.pkl", "label_cls.pkl", "text_cls.pkl"]
for rel in required:
    assert (base / rel).exists(), f"missing {rel}"

samples = pickle.load(open(base / "samples.pkl", "rb"))
relevance_map = pickle.load(open(base / "relevance_map.pkl", "rb"))
label_cls = pickle.load(open(base / "label_cls.pkl", "rb"))
text_cls = pickle.load(open(base / "text_cls.pkl", "rb"))

assert len(samples) == 806791, len(samples)
assert len(relevance_map) == 798385, len(relevance_map)
assert len(label_cls) == 103, len(label_cls)
assert len(text_cls) == 798385, len(text_cls)
assert len(set(label for sample in samples for label in sample["labels"])) == 103
assert len(set(sample["text_idx"] for sample in samples)) == 798385

all_ids = set(range(len(samples)))
for fold in range(5):
    split_sets = {}
    for split in ["train", "val", "test"]:
        path = base / f"fold_{fold}" / f"{split}.pkl"
        assert path.exists(), f"missing {path}"
        split_sets[split] = set(pickle.load(open(path, "rb")))

    assert not (split_sets["train"] & split_sets["val"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["val"] & split_sets["test"])
    assert split_sets["train"] | split_sets["val"] | split_sets["test"] == all_ids

print("RCV1-103-H3 dataset OK")
PY
```

## 4. Gerar `target_descriptions.pkl` com OpenAI API

Configure a chave:

```bash
export OPENAI_API_KEY="sk-..."
```

Primeiro rode dry-run para estimar custo e validar amostragem:

```bash
python generate_target_descriptions.py \
  --dataset RCV1-103-H3 \
  --model gpt-4o \
  --max_samples 256 \
  --sampling_strategy uniform \
  --dry_run
```

Depois rode via Batch API:

```bash
python generate_target_descriptions.py \
  --dataset RCV1-103-H3 \
  --model gpt-4o \
  --max_samples 256 \
  --sampling_strategy uniform \
  --num_batches 1
```

Alternativa sincrona:

```bash
python generate_target_descriptions.py \
  --dataset RCV1-103-H3 \
  --model gpt-4o \
  --max_samples 256 \
  --sampling_strategy uniform \
  --sync
```

Validar que as chaves batem com os labels reais do dataset:

```bash
python - <<'PY'
import pickle
from pathlib import Path

samples = pickle.load(open("resource/dataset/RCV1-103-H3/samples.pkl", "rb"))
labels = set(label for sample in samples for label in sample["labels"])

target_path = Path("resource/llm/RCV1-103-H3/target_descriptions.pkl")
target = pickle.load(open(target_path, "rb"))

assert len(target) == 103, len(target)
assert set(target) == labels, f"missing={labels - set(target)} extra={set(target) - labels}"

print("target_descriptions.pkl OK")
PY
```

## 5. Subir vLLM local

Em outro terminal:

```bash
cd /root/RAG-Fuse-Flavio

eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate RAG-Fuse

python -m vllm.entrypoints.openai.api_server \
  --model /home/datalake/models/llama3-8b \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45
```

Validar:

```bash
curl http://localhost:8001/v1/models
```

## 6. Rodar prompt optimization

```bash
python main.py \
  tasks=[prompt_opt] \
  data=RCV1-103-H3 \
  data.text_features_source=TXT \
  llm.server.base_url=http://localhost:8001/v1 \
  llm.server.model=/home/datalake/models/llama3-8b
```

## 7. Criar RAG-labels por fold

```bash
for fold in $(seq 0 4); do
  python main.py \
    tasks=[label_desc] \
    data=RCV1-103-H3 \
    model=RetrieverRoBERTa \
    data.folds=[$fold] \
    data.text_features_source=TXT \
    llm.server.base_url=http://localhost:8001/v1 \
    llm.server.model=/home/datalake/models/llama3-8b
done
```

Validar:

```bash
python - <<'PY'
import pickle
from pathlib import Path

for fold in range(5):
    path = Path(f"resource/dataset/RCV1-103-H3/fold_{fold}/labels_descriptions.pkl")
    assert path.exists(), f"missing {path}"
    labels_desc = pickle.load(open(path, "rb"))
    assert len(labels_desc) == 103, f"{path}: {len(labels_desc)}"

print("RAG-labels OK")
PY
```

## 8. Rodar pipeline principal

```bash
bash run.sh RCV1-103-H3 0 4
```

## 9. Validar outputs finais

```bash
find resource/ranking -maxdepth 2 -type f | grep 'RCV1-103-H3' | sort
find resource/result -maxdepth 2 -type f | grep 'RCV1-103-H3' | sort
find resource/model_checkpoint -type f | grep 'RCV1-103-H3' | sort
```

Outputs esperados incluem:
- `resource/ranking/BM25_RCV1-103-H3/`
- `resource/ranking/LLM_RetrieverRoBERTa_RCV1-103-H3/`
- `resource/ranking/Fused_LLM_RetrieverRoBERTa_RCV1-103-H3/`
- `resource/ranking/Aggregated_LLM_RetrieverRoBERTa_RCV1-103-H3/`
- `resource/result/*RCV1-103-H3*/`
- `resource/model_checkpoint/LLM_RetrieverRoBERTa_RCV1-103-H3_*.ckpt`

## Observacoes

- Nao reutilize `resource/llm/RCV1/target_descriptions.pkl`: ele tem chaves antigas como `C11`, que nao batem com os labels textuais do golden truth.
- O fluxo OpenAI gera apenas `target_descriptions.pkl`; `prompt_opt` e `label_desc` usam o vLLM local.
- Se `bash run.sh RCV1-103-H3 0 4` falhar por memoria, reduza `data.batch_size=128` para `64` dentro de `run/RCV1-103-H3.sh`.
