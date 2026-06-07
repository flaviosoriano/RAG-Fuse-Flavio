# Rodar RAG-Fuse em `LBD-UFMG/WOS-150-H2`

Este roteiro prepara e executa o RAG-Fuse no dataset `LBD-UFMG/WOS-150-H2`, usando o nome local `WOS-150-H2`.

Arquivos estaticos preparados no repo:
- `setting/data/WOS-150-H2.yaml`
- `run/WOS-150-H2.sh`
- `resource/llm/WOS-150-H2/seed_prompt.txt`
- `resource/llm/WOS-150-H2/meta_prompt.txt`

Observado no dataset remoto:
- 46.985 amostras/textos
- 150 labels
- 5 folds: `0..4`
- 2 labels relevantes por texto, preservando a hierarquia pai + H2
- artifacts prontos: `samples.pkl`, `relevance_map.pkl`, `label_cls.pkl`, `text_cls.pkl`, `fold_*/train.pkl`, `fold_*/val.pkl`, `fold_*/test.pkl`

Nao rode `0 9`: este dataset tem somente folds `0..4`.

## 1. Preparar ambiente

```bash
cd /root/RAG-Fuse-Flavio

eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate RAG-Fuse
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Se ainda nao estiver autenticado no Hugging Face:

```bash
/root/miniconda3/envs/RAG-Fuse/bin/hf auth login
```

## 2. Baixar dataset golden truth

```bash
/root/miniconda3/envs/RAG-Fuse/bin/hf download LBD-UFMG/WOS-150-H2 \
  --repo-type dataset \
  --local-dir resource/dataset/WOS-150-H2
```

## 3. Validar formato do dataset

```bash
python - <<'PY'
import pickle
from collections import Counter
from pathlib import Path

base = Path("resource/dataset/WOS-150-H2")

required = ["samples.pkl", "relevance_map.pkl", "label_cls.pkl", "text_cls.pkl"]
for rel in required:
    assert (base / rel).exists(), f"missing {rel}"

samples = pickle.load(open(base / "samples.pkl", "rb"))
relevance_map = pickle.load(open(base / "relevance_map.pkl", "rb"))
label_cls = pickle.load(open(base / "label_cls.pkl", "rb"))
text_cls = pickle.load(open(base / "text_cls.pkl", "rb"))

assert len(samples) == 46985, len(samples)
assert len(relevance_map) == 46985, len(relevance_map)
assert len(label_cls) == 150, len(label_cls)
assert len(text_cls) == 46985, len(text_cls)
assert len(set(sample["text_idx"] for sample in samples)) == 46985
assert len(set(label for sample in samples for label in sample["labels"])) == 150
assert Counter(len(sample["labels_ids"]) for sample in samples) == {2: 46985}
assert Counter(len(labels) for labels in relevance_map.values()) == {2: 46985}

all_ids = set(range(len(samples)))
for fold in range(5):
    split_sets = {}
    for split in ["train", "val", "test"]:
        path = base / f"fold_{fold}" / f"{split}.pkl"
        assert path.exists(), f"missing {path}"
        split_sets[split] = set(pickle.load(open(path, "rb")))

    assert len(split_sets["train"]) == 33829
    assert len(split_sets["val"]) == 3759
    assert len(split_sets["test"]) == 9397
    assert not (split_sets["train"] & split_sets["val"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["val"] & split_sets["test"])
    assert split_sets["train"] | split_sets["val"] | split_sets["test"] == all_ids

print("WOS-150-H2 dataset OK")
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
  --dataset WOS-150-H2 \
  --model gpt-4o \
  --max_samples 256 \
  --sampling_strategy uniform \
  --dry_run
```

Depois rode via Batch API:

```bash
python generate_target_descriptions.py \
  --dataset WOS-150-H2 \
  --model gpt-4o \
  --max_samples 256 \
  --sampling_strategy uniform \
  --num_batches 1
```

Alternativa sincrona:

```bash
python generate_target_descriptions.py \
  --dataset WOS-150-H2 \
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

samples = pickle.load(open("resource/dataset/WOS-150-H2/samples.pkl", "rb"))
labels = set(label for sample in samples for label in sample["labels"])

target_path = Path("resource/llm/WOS-150-H2/target_descriptions.pkl")
target = pickle.load(open(target_path, "rb"))

assert len(target) == 150, len(target)
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
  data=WOS-150-H2 \
  data.text_features_source=TXT \
  llm.server.base_url=http://localhost:8001/v1 \
  llm.server.model=/home/datalake/models/llama3-8b
```

## 7. Criar RAG-labels por fold

```bash
for fold in $(seq 0 4); do
  python main.py \
    tasks=[label_desc] \
    data=WOS-150-H2 \
    model=RetrieverBERT \
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
    path = Path(f"resource/dataset/WOS-150-H2/fold_{fold}/labels_descriptions.pkl")
    assert path.exists(), f"missing {path}"
    labels_desc = pickle.load(open(path, "rb"))
    assert len(labels_desc) == 150, f"{path}: {len(labels_desc)}"

print("RAG-labels OK")
PY
```

## 8. Rodar pipeline principal

Este script executa:

1. `sparse_retrieve`
2. `fit`
3. `predict`
4. `eval`
5. `fuse`
6. `aggregate`

```bash
bash run.sh WOS-150-H2 0 4
```

Se precisar reduzir workers:

```bash
DENSE_NUM_WORKERS=4 bash run.sh WOS-150-H2 0 4
```

## 9. Validar outputs finais

```bash
find resource/ranking -maxdepth 2 -type f | grep 'WOS-150-H2' | sort
find resource/result -maxdepth 2 -type f | grep 'WOS-150-H2' | sort
find resource/model_checkpoint -type f | grep 'WOS-150-H2' | sort
```

Outputs esperados incluem:
- `resource/ranking/BM25_WOS-150-H2/`
- `resource/ranking/LLM_RetrieverBERT_WOS-150-H2/`
- `resource/ranking/Fused_LLM_RetrieverBERT_WOS-150-H2/`
- `resource/ranking/Aggregated_LLM_RetrieverBERT_WOS-150-H2/`
- `resource/result/*WOS-150-H2*/`
- `resource/model_checkpoint/LLM_RetrieverBERT_WOS-150-H2_*.ckpt`

## Observacoes

- A config preserva `num_relevant_labels: 2`, porque o dataset remoto codifica label pai + label H2 para cada texto.
- Nao reutilize descricoes de outro dataset; gere `resource/llm/WOS-150-H2/target_descriptions.pkl`.
- A etapa `label_desc` grava `labels_descriptions.pkl` dentro de cada `fold_*`, que e obrigatorio para `label_enhancement=LLM`.
- O script exporta `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1` durante o pipeline principal, seguindo o padrao de `RCV1-103-H3.sh`. Garanta que `bert-base-uncased` ja esteja no cache local antes de rodar.
