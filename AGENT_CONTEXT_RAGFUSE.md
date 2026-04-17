# AGENT_CONTEXT_RAGFUSE

This file consolidates the operational, conceptual, and experimental context of the **RAG-Fuse** project, based on two primary sources:

1. the RAG-Fuse method paper;
2. the inspection of the current local repository code.

## Scope and source-of-truth principle

- **Source of truth for system behavior:** current repository implementation.
- **Source of truth for conceptual motivation and design:** RAG-Fuse paper.
- **README, auxiliary scripts, and historical naming** must be treated as secondary evidence whenever they diverge from the code.
- Whenever a conclusion below is deduced rather than directly observed, it must be treated as **inference**.

---

## 1. Executive summary

RAG-Fuse reformulates text classification as a **label ranking** problem. Instead of training a traditional classifier that directly emits logits over classes, the method treats:

- the **text** as a **query**;
- the **classes** as **documents/candidates**;
- the final decision as the result of **retrieval + fusion + aggregation**.

In the paper, the core proposal combines:

- a **sparse retriever**, focused on lexical signals;
- a **dense retriever**, focused on semantic signals;
- **RAG-labels**, which semantically enrich class representations;
- a **ranking fusion** stage;
- a final **aggregation** stage over `head` and `tail` classes to mitigate imbalance.

In the current implementation observed in the workspace, the main live pipeline is composed of six blocks:

1. `sparse_retrieve`
2. `fit`
3. `predict`
4. `eval`
5. `fuse`
6. `aggregate`

In addition, there is an auxiliary LLM flow for:

- generating `target_descriptions.pkl`;
- optimizing prompts;
- generating `labels_descriptions.pkl` by fold.

However, repository inspection suggests that this LLM block is **partially misaligned** with the main flow and requires extra caution before treating it as robust.

---

## 2. What RAG-Fuse is, conceptually

### 2.1 Target problem

The paper positions RAG-Fuse for text classification scenarios with:

- **high lexical-semantic gap** between texts and class labels;
- **severe class imbalance**;
- difficulty for traditional classifiers and even more expensive neural approaches.

The central motivation is that, in many cases, the document text and the class name do not share enough direct vocabulary. Instead of learning a conventional classifier only, the method measures **relevance** between text and class as a retrieval problem.

### 2.2 Retrieval reformulation

In the paper, the system is described as a pipeline in which:

- the text enters as a query;
- the class space becomes a search universe;
- two retrievers generate partial rankings;
- fusion and aggregation produce the final label ranking.

### 2.3 Role of RAG-labels

**RAG-labels** are enriched class descriptions generated with the support of an LLM plus retrieved context, with the purpose of reducing the gap between:

- long, context-rich texts;
- short, ambiguous, semantically poor labels.

In the paper’s design, this is a core component of the proposal, not a peripheral detail.

### 2.4 Role of head/tail

The paper segments the class space into **majority** and **minority** subsets, often referred to as `H` and `T`, to reduce bias toward frequent classes. Fusion and aggregation are performed respecting this separation.

---

## 3. Summary of what the current code implements

Inspection of the local code shows that the project implements a label retrieval pipeline, not a pure text->class classifier. The current experimental system combines:

- sparse retrieval with BM25;
- dense dual-encoder training with BERT/RoBERTa;
- text and label representation prediction;
- dense evaluation via HNSW (`nmslib`);
- fusion of BM25 + dense rankings;
- final aggregation of `head` + `tail` rankings, including propensity-scored metrics.

The repository also contains an LLM flow for generating label descriptions, but the current state suggests it is not as operationally robust as the main pipeline.

---

## 4. End-to-end mental model of the flow

### 4.1 Reliable main flow

The most stable operational path observed is:

1. `run.sh` receives the dataset and fold interval;
2. it delegates to `run/<DATASET>.sh`;
3. these scripts execute separate stages by fold via `python main.py ...`;
4. `main.py` resolves Hydra configuration and dispatches tasks to specific helpers.

The resulting flow is:

1. `sparse_retrieve`
2. `fit`
3. `predict`
4. `eval`
5. `fuse`
6. `aggregate`

### 4.2 Auxiliary LLM flow

There is also a secondary flow:

1. `generate_target_descriptions.py`
2. `prompt_opt`
3. `label_desc`

**Important inference:** the flow that appears most faithful to current operational use of the repository is the main retrieval/fusion/aggregation pipeline. The LLM flow seems to exist as an important extension, but still with fragility in compatibility and orchestration.

---

## 5. Project data contract

The data contract revolves around pickle-serialized artifacts, with central role for:

- `samples.pkl`
- `relevance_map.pkl`
- `label_cls.pkl`
- `text_cls.pkl`
- `train.pkl`, `val.pkl`, `test.pkl` per `fold_<N>`

### 5.1 Expected dataset structure

Expected directory:

```text
resource/dataset/<DATASET>/
  samples.pkl
  relevance_map.pkl
  label_cls.pkl
  text_cls.pkl
  fold_<N>/
    train.pkl
    val.pkl
    test.pkl
    labels_descriptions.pkl   # when label_enhancement=LLM
    <pseudo_labels>.pkl       # when label_enhancement=PMI
```

### 5.2 `samples.pkl`

Observed structure:

```python
[
    {
        "idx": int,
        "text_idx": int,
        "text": str,
        "labels_ids": list[int],
        "labels": list[str],
    },
    ...
]
```

Practical meaning:

- `idx`: serialized sample identifier;
- `text_idx`: text identifier for ranking/evaluation;
- `text`: main textual content;
- `labels_ids`: numeric label ids;
- `labels`: textual forms of labels.

### 5.3 `relevance_map.pkl`

Expected structure:

```python
{text_idx: [label_idx, ...]}
```

Observed use:

- loss miner;
- validation metric;
- dense evaluation;
- BM25 baseline;
- final aggregation.

### 5.4 `label_cls.pkl`

Expected structure:

```python
{label_idx: ["all", "head"]}  # or ["all", "tail"]
```

This file controls the partition between frequent and tail labels.

### 5.5 `text_cls.pkl`

Expected structure:

```python
{text_idx: ["all", "head"]}    # or ["all", "tail"] or both
```

This file controls in which ranking subsets a text participates. In multi-label scenarios, the same `text_idx` may be associated with both `head` and `tail`.

### 5.6 Fold splits

`train.pkl`, `val.pkl`, and `test.pkl` are lists of `idx`, not `text_idx`. This matters because:

- splitting occurs over serialized samples;
- final evaluation occurs at the `text_idx` level.

### 5.7 `labels_descriptions.pkl`

Expected structure:

```python
{label_idx: "generated description"}
```

This artifact is used when `label_enhancement=LLM`.

---

## 6. Correct interpretation of `idx` vs `text_idx`

This distinction is central to any correct understanding of the repository.

### 6.1 `idx`
- identifies the serialized sample;
- is used by fold splits.

### 6.2 `text_idx`
- identifies the text at the ranking/evaluation level;
- is the main unit for relevance and final metrics.

### 6.3 Consequence
There may be duplicated `text_idx` values across multiple serialized samples. The pipeline appears to tolerate this, but it may increase cost and make the contract less intuitive.

---

## 7. Main flow detailed by stage

## 7.1 `sparse_retrieve`

### Role
Runs the sparse baseline/retrieval branch, based on BM25, projecting retrieved documents into candidate labels.

### Input
- `samples.pkl`
- splits
- `label_cls.pkl`
- `text_cls.pkl`
- `relevance_map.pkl`

### Output
- BM25 ranking by `head`/`tail`
- partial `.rts` result file

### Interpretation
This component is the code-level realization of the sparse retriever described in the paper.

---

## 7.2 `fit`

### Role
Trains the dense dual-encoder retriever.

### Mental model
- training expands samples into `(text, label)` pairs;
- the model learns a shared vector space between texts and labels;
- the objective is to bring relevant pairs closer and push irrelevant ones apart.

### Output
- `.ckpt` checkpoint
- training logs

### Interpretation
This is the core of the dense retriever described in the paper.

---

## 7.3 `predict`

### Role
Generates embeddings for all texts and labels, saved as `.prd` batches.

### Output
Files in:

```text
resource/prediction/<MODEL_NAME>_<DATASET>/fold_<N>/<DATALOADER_IDX>_<BATCH_IDX>.prd
```

### Interpretation
This stage decouples training from dense retrieval evaluation.

---

## 7.4 `eval`

### Role
Executes dense label retrieval using previously generated embeddings and an HNSW index.

### Operation
- loads `.prd`;
- filters texts/labels for the target split;
- builds an HNSW index over labels;
- retrieves nearest labels per text;
- saves dense rankings by `head` and `tail`.

### Interpretation
This is the code-level implementation of dense retrieval.

---

## 7.5 `fuse`

### Role
Combines BM25 ranking with dense ranking, separately for `head` and `tail`, using `ranx.fuse(norm="zmuv", method="mnz")`.

### Output
```text
resource/ranking/Fused_<MODEL_NAME>_<DATASET>/Fused_<MODEL_NAME>_<DATASET>_<FOLD>.rnk
```

### Interpretation
This directly corresponds to the fusion stage described in the paper.

---

## 7.6 `aggregate`

### Role
Merges `head` and `tail` rankings into a single final ranking per text and computes final metrics, including propensity-scored ones.

### Output
```text
resource/ranking/Aggregated_<MODEL_NAME>_<DATASET>/Aggregated_<MODEL_NAME>_<DATASET>_<FOLD>.rnk
```

and final `.rts` result files.

### Interpretation
This is the final stage of the pipeline, corresponding to the paper’s aggregation stage.

---

## 8. Current logical architecture of the project

### 8.1 Orchestration
- `run.sh` selects the operational pipeline;
- `run/<DATASET>.sh` defines actual overrides by dataset/fold;
- `main.py` maps Hydra tasks to concrete helpers.

### 8.2 Configuration
- Hydra combines `setting/setting.yaml` with `setting/data/<DATASET>.yaml` and `setting/model/<MODEL>.yaml`;
- effective configuration is almost never only the base YAML: it is **YAML + shell overrides**.

### 8.3 Data preparation
- the pipeline assumes pickle-serialized data;
- `samples.pkl` is the center of the structural contract;
- `labels_descriptions.pkl` is optional, but required in `LLM` mode.

### 8.4 Modeling
- main model: symmetric dual-encoder;
- active encoders: BERT or RoBERTa;
- pooling: concatenation of last 4 layers and normalized initial token;
- active loss: NT-Xent from `pytorch-metric-learning` with inner-product distance.

### 8.5 Training
- each sample becomes one or more positive `(text, label)` pairs;
- the miner uses `relevance_map.pkl`;
- early stopping and checkpointing monitor `val_MRR`.

### 8.6 Prediction
- there are two dataloaders: texts and labels;
- `RetrieverPredictionWriter` saves `.prd` batches.

### 8.7 Evaluation
- dense evaluation uses HNSW over labels;
- sparse evaluation uses BM25 over labeled texts.

### 8.8 Fusion and aggregation
- fusion combines BM25 and dense rankings;
- aggregation combines `head` and `tail`;
- propensity-scored metrics enter in this stage.

---

## 9. Map of critical files

## 9.1 Entrypoints and orchestration

### `main.py`
- central dispatcher of Hydra tasks;
- first file to inspect when altering flow.

### `run.sh`
- operational entrypoint for full experiments.

### `run/*.sh`
- contain the actual overrides used in experiments by dataset/fold;
- often more revealing of effective runtime behavior than the base YAML files.

## 9.2 Configuration

### `setting/setting.yaml`
- global Hydra configuration;
- especially important because defaults may induce surprising behavior.

### `setting/data/*.yaml`
- dataset-specific contract and hyperparameters.

### `setting/model/*.yaml`
- define backbone, tokenizer, loss, and encoder hyperparameters.

## 9.3 Data and dataloaders

### `source/datamodule/RetrieverDataModule.py`
- builds datasets and loaders for training/prediction.

### `source/dataset/RetrieverFitDataset.py`
- expands samples into `(text, label)` pairs.

### `source/dataset/TextDataset.py`
- prepares text inputs for inference.

### `source/dataset/LabelDataset.py`
- deduplicates labels and builds enriched label inputs.

## 9.4 Modeling

### `source/model/RetrieverModel.py`
- LightningModule effectively used in the main pipeline.

### `source/encoder/*.py`
- backbone adapters and HF interfaces.

### `source/pooling/ConcatenatePooling.py`
- defines the final embedding.

### `source/loss/RetrieverLoss.py`
- active loss of the model.

### `source/miner/RelevanceMiner.py`
- constructs positives/negatives per batch from `relevance_map.pkl`.

### `source/metric/RetrieverMetric.py`
- validation MRR metric.

## 9.5 Evaluation, fusion, and aggregation

### `source/helper/SparseRetrieverHelper.py`
- BM25 and document->label projection.

### `source/helper/retriever/RetrieverFitHelper.py`
- orchestrates Lightning training.

### `source/helper/retriever/RetrieverPredictHelper.py`
- orchestrates prediction and `.prd` writing.

### `source/helper/retriever/RetrieverEvalHelper.py`
- performs dense retrieval via HNSW and produces rankings.

### `source/helper/RankingFusionHelper.py`
- implements BM25 + dense ensemble.

### `source/helper/RankingAggregationHelper.py`
- aggregates head/tail and computes final metrics.

## 9.6 LLM flow

### `generate_target_descriptions.py`
- generates `target_descriptions.pkl` via OpenAI.

### `source/helper/PromptOptimizerHelper.py`
- optimizes prompts based on embedding similarity.

### `source/helper/LabelDescriptionHelper.py`
- generates `labels_descriptions.pkl` per fold.

### `source/llm/__init__.py`
- async client for an OpenAI-compatible vLLM server.

---

## 10. What is clearly implemented vs what is fragile

## 10.1 Strongly evidenced as implemented

- sparse retrieval with BM25;
- dense dual-encoder retrieval;
- generation of text and label embeddings;
- ANN/HNSW retrieval over labels;
- fusion of BM25 + dense rankings;
- final head/tail aggregation;
- use of propensity-scored metrics.

## 10.2 Partially implemented / fragile

- prompt optimization flow;
- operational generation of `labels_descriptions.pkl` by fold;
- robust integration of the LLM block into the main pipeline.

## 10.3 Evidence of misalignment

Repository inspection points to at least the following fragilities:

1. **`label_desc` appears inconsistent with existing `optimized_prompt.txt` files**
   - `LabelDescriptionHelper` expects certain placeholders;
   - existing prompts use different placeholders;
   - probable result: `KeyError` during execution.

2. **Naming inconsistencies across training/prediction/evaluation and fusion/aggregation**
   - some scripts use different model names across stages;
   - if expected artifacts do not exist under the exact downstream name, the pipeline breaks.

3. **Surprising default in `setting/setting.yaml`**
   - running `python main.py` without overrides may start `label_desc`, not training/evaluation.

---

## 11. Important risks and pitfalls

## 11.1 High risk

### A. `label_desc` may break because of placeholder incompatibility
This is one of the clearest operational fragilities.

### B. Artifact naming may diverge across stages
The pipeline may fail simply because upstream and downstream names do not match.

### C. Hydra default tasks may induce the wrong execution path
Running `main.py` without explicit overrides may launch an unexpected task.

## 11.2 Medium risk

### D. Training validation may not use exactly `val.pkl`
Inspection suggests training validation may rely on a slice of training data rather than the formal fold validation split. This can affect early stopping and comparability.

### E. Real evaluation focuses on the `test` split
Even if generic functions exist, the operational flow appears to persist mainly test results.

### F. `label_desc` reuses hyperparameters from `prompt_opt`
The `llm.label_desc` YAML block may not truly control runtime behavior as one might expect.

### G. `KWD` exists in code but not in the observed data contract
There is code support for `sample["keywords"]`, but observed `samples.pkl` files do not seem to materialize that field.

## 11.3 Structural technical debt

- presence of partially dead or disconnected modules;
- divergence between README and code in some contract-level details;
- destructive `reset_resource.sh` script with risk of misuse;
- duplicated `text_idx` increasing cost and opacity.

---

## 12. How to interpret README, paper, and code

### Operational rule
If there is conflict:

1. **current repository code**
2. **paper**
3. **README / old scripts / historical conventions**

### Rationale
- the paper explains the intended method;
- the code determines current reproducible behavior;
- the README may be stale or oversimplify critical details.

---

## 13. How the agent should think about the project

The agent using this context must treat RAG-Fuse as:

- a **retrieval system for label assignment**;
- a hybrid pipeline combining **lexical** and **semantic** signals;
- a system sensitive to **imbalance** and `head`/`tail` partitioning;
- a project where the **main flow** is more mature than the **auxiliary LLM flow**.

---

## 14. Correct mode of answering about the project

Whenever analyzing or answering about RAG-Fuse, the agent should organize its reasoning in one or more of the following layers:

### 14.1 Conceptual layer
- what method hypothesis is involved?
- what part of the paper does this represent?

### 14.2 Implementation layer
- which file/helper/model does this today?
- is the implementation mainline, auxiliary, or incomplete?

### 14.3 Experimental layer
- how should the effect be measured?
- what main metric makes sense?
- what baseline should be preserved?

### 14.4 Risk layer
- can this change break compatibility?
- are there dependent intermediate artifacts?
- could the hypothesis be confused with naming, split logic, caching, or prompt issues?

---

## 15. Honesty and rigor rules

The agent must always:

- distinguish **observed fact** from **inference**;
- never claim something is “implemented” without grounding in code/context;
- avoid proposing major changes without indicating **affected files**, **risk**, and **minimum test**;
- treat reproducibility as a first-class objective;
- remember that the LLM flow, while promising, is currently more fragile.

---

## 16. Questions this context should make the agent strong at

This context is designed to make the agent particularly strong at answering questions such as:

- “Explain RAG-Fuse from the paper and show where it appears in the code.”
- “What is the truly reliable main flow of the project?”
- “Where is the fusion between BM25 and dense retriever implemented?”
- “Is the use of RAG-labels actually operational today?”
- “Which files do I need to modify to change the dense retriever?”
- “If I want to improve the method scientifically, what minimum experiment should I run?”
- “What are the biggest risks of changing the pipeline without breaking reproducibility?”

---

## 17. Final synthesis

In its current form, the project should be understood as follows:

- **Conceptually**, RAG-Fuse is a text classification approach via retrieval and label ranking, with semantic class enrichment and fusion of lexical and dense evidence.
- **Implementation-wise**, the main live pipeline is BM25 -> dense training -> prediction -> HNSW evaluation -> fusion -> aggregation.
- **Experimentally**, the project strongly depends on a data contract centered on `samples.pkl`, relevance maps, and `head`/`tail` partitioning.
- **Operationally**, the LLM block exists, but must currently be treated with caution due to prompt, configuration, and naming inconsistencies.

Therefore, any specialist agent for RAG-Fuse should act as:
- interpreter of the paper;
- rigorous reader of the code;
- auditor of experimental compatibility;
- proposer of improvements under scientific discipline.
