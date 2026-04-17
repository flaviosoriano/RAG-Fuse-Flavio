# WOS-46985 no RAG-Fuse: estado atual e próximos passos

Este documento registra o que já foi feito para onboard do dataset `WOS-46985` no projeto e o que ainda falta para concluir a integração e o teste ponta a ponta.

## Objetivo

Preparar o `WOS-46985` a partir da fonte oficial, adequá-lo ao contrato de dados do RAG-Fuse, deixar uma cópia pronta para publicação no Hugging Face e integrar o dataset ao pipeline principal.

## Fonte validada

- Fonte canônica validada: `https://data.mendeley.com/datasets/9rw3vkcfy4/2`
- DOI validado: `10.17632/9rw3vkcfy4.2`
- Licença validada: `CC BY 4.0`
- Espelho usado apenas para verificação: `https://huggingface.co/datasets/HDLTex/web_of_science`
- Endpoint público do archive usado na prática:
  `https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded`

## O que já foi feito

### 1. Validação do conteúdo oficial

O archive oficial foi baixado e inspecionado. Ele contém:

- `WOS46985/X.txt`
- `WOS46985/Y.txt`
- `WOS46985/YL1.txt`
- `WOS46985/YL2.txt`
- `Meta-data/Data.xlsx`

Também foi confirmado que:

- `X/Y/YL1/YL2` têm `46.985` linhas
- `Y` tem `134` valores únicos
- `YL1` tem `7` valores únicos
- `YL2` tem `53` valores únicos
- `Data.xlsx` tem os campos `Y1`, `Y2`, `Y`, `Domain`, `area`, `keywords`, `Abstract`

Observação importante:

- `YL2` não é identificador flat globalmente único
- o metadado oficial tem `8` ids `Y` com mais de um `area` bruto
- a preparação resolveu isso por maioria e registrou as alternativas no `metadata.json`

### 2. Script de preparo do dataset

Foi criado o arquivo [prepare_wos_46985.py](/home/flaviossf/work/benchmark-celso/RAG-Fuse/prepare_wos_46985.py).

Esse script faz:

- download do archive oficial
- extração do `WOS46985`
- cópia da fonte crua para `resource/dataset/raw_dataset/WOS-46985/`
- leitura de `Meta-data/Data.xlsx`
- construção do `label_text_map` a partir de `Y -> area`
- geração dos artefatos no formato esperado pelo RAG-Fuse
- geração de pasta de staging para upload ao Hugging Face

### 3. Integração ao pipeline principal

Foram criados:

- [setting/data/WOS-46985.yaml](/home/flaviossf/work/benchmark-celso/RAG-Fuse/setting/data/WOS-46985.yaml)
- [run/WOS-46985.sh](/home/flaviossf/work/benchmark-celso/RAG-Fuse/run/WOS-46985.sh)

Configuração adotada nesta primeira integração:

- `label_enhancement: NONE`
- `text_features_source: TXT`
- `model: RetrieverBERT`
- fluxo principal: `sparse_retrieve -> fit -> predict -> eval -> fuse -> aggregate`

### 4. Script de publicação para o Hugging Face

Foi criado o arquivo [publish_wos_46985_to_hf.py](/home/flaviossf/work/benchmark-celso/RAG-Fuse/publish_wos_46985_to_hf.py).

Esse script faz:

- leitura do token local (`HF_TOKEN`, `HUGGINGFACE_TOKEN` ou login salvo)
- criação do dataset repo no Hub
- upload da pasta de staging pronta

### 5. Dataset já materializado localmente

Fonte crua já foi copiada para:

- `resource/dataset/raw_dataset/WOS-46985/X.txt`
- `resource/dataset/raw_dataset/WOS-46985/Y.txt`
- `resource/dataset/raw_dataset/WOS-46985/YL1.txt`
- `resource/dataset/raw_dataset/WOS-46985/YL2.txt`
- `resource/dataset/raw_dataset/WOS-46985/Meta-data/Data.xlsx`
- `resource/dataset/raw_dataset/WOS-46985/SOURCE.json`

Artefatos processados já foram gerados em:

- `resource/dataset/WOS-46985/samples.pkl`
- `resource/dataset/WOS-46985/relevance_map.pkl`
- `resource/dataset/WOS-46985/label_cls.pkl`
- `resource/dataset/WOS-46985/text_cls.pkl`
- `resource/dataset/WOS-46985/labels_map.pkl`
- `resource/dataset/WOS-46985/metadata.json`
- `resource/dataset/WOS-46985/README.md`
- `resource/dataset/WOS-46985/fold_0..9/{train,val,test}.pkl`

### 6. Staging do Hugging Face já foi gerado

A pasta pronta para upload já existe em:

- `resource/dataset/hf_staging/wos-46985-rag-fuse/README.md`
- `resource/dataset/hf_staging/wos-46985-rag-fuse/raw/original/...`
- `resource/dataset/hf_staging/wos-46985-rag-fuse/processed/rag_fuse/...`

## O que ainda falta fazer

### 1. Atualizar a seção `WOS-46985` do notebook

Status: pendente.

O notebook [RAGFuse_Datasets.ipynb](/home/flaviossf/work/benchmark-celso/RAG-Fuse/RAGFuse_Datasets.ipynb) ainda aponta para:

- `variant_dir = Path("resource/dataset/WOS-46985")`

Ele ainda não foi reescrito para:

- usar `resource/dataset/raw_dataset/WOS-46985/` como fonte crua
- ler `Meta-data/Data.xlsx`
- preencher `samples.pkl["labels"]` com nomes textuais
- salvar `labels_map.pkl`
- gravar `metadata.json` corrigido com `dataset_name: "WOS-46985"`

Esse é o principal ponto funcional ainda em aberto no repositório.

### 2. Publicar no Hugging Face

Status: pendente por autenticação.

Foi verificado que o ambiente local **não** tem token do Hugging Face disponível neste momento.

Resultado:

- o script de upload foi criado
- a pasta de staging já está pronta
- o upload em si ainda não foi executado

### 3. Rodar smoke test do pipeline

Status: pendente.

Ainda falta validar pelo menos o fold `0` com:

- `sparse_retrieve`
- `fit`
- `predict`
- `eval`
- `fuse`
- `aggregate`

### 4. Rodar benchmark completo

Status: pendente.

Depois do smoke test do fold `0`, ainda falta executar o range completo `0..9`.

## Passo a passo recomendado daqui para frente

### Passo 1. Revisar rapidamente os artefatos já gerados

Comandos úteis:

```bash
find resource/dataset/raw_dataset/WOS-46985 -maxdepth 3 -type f | sort
find resource/dataset/WOS-46985 -maxdepth 2 -type f | sort
sed -n '1,220p' resource/dataset/WOS-46985/metadata.json
```

### Passo 2. Atualizar manualmente o notebook

Você precisa ajustar a seção final `WOS-46985` de [RAGFuse_Datasets.ipynb](/home/flaviossf/work/benchmark-celso/RAG-Fuse/RAGFuse_Datasets.ipynb) para refletir o fluxo novo.

Checklist do que alterar no notebook:

1. Trocar a fonte crua de `resource/dataset/WOS-46985` para `resource/dataset/raw_dataset/WOS-46985`
2. Ler `Meta-data/Data.xlsx`
3. Construir `label_text_map` usando `Y -> area`
4. Usar `int(Y)` como `label_id`, sem `factorize`
5. Preencher `labels` com os nomes textuais resultantes
6. Salvar também `labels_map.pkl`
7. Corrigir `metadata.json` para `dataset_name: "WOS-46985"`
8. Registrar no notebook que `YL2` não é id flat global

### Passo 3. Publicar no Hugging Face

Primeiro autentique:

```bash
huggingface-cli login
```

Ou exporte o token:

```bash
export HF_TOKEN=...
```

Depois rode:

```bash
conda run -n rag_fuse python publish_wos_46985_to_hf.py \
  --repo-id <seu-usuario>/wos-46985-rag-fuse
```

Se quiser reprocessar tudo antes do upload:

```bash
conda run -n rag_fuse python prepare_wos_46985.py --repo-root "$(pwd)"
```

Observação:

- eu executei o preparo reaproveitando o archive já baixado em `/tmp`
- o downloader do script foi corrigido para usar `User-Agent`, mas o fluxo completo de download automático ainda não foi revalidado depois dessa correção

### Passo 4. Rodar smoke test do fold 0

Comandos sugeridos:

```bash
conda run -n rag_fuse python main.py \
  tasks=[sparse_retrieve] \
  model=BM25 \
  data=WOS-46985 \
  data.text_features_source=TXT \
  data.folds=[0]
```

```bash
conda run -n rag_fuse python main.py \
  tasks=[fit] \
  trainer.max_epochs=1 \
  trainer.patience=1 \
  model=RetrieverBERT \
  model.name=NONE_RetrieverBERT \
  data=WOS-46985 \
  data.text_max_length=256 \
  data.label_max_length=16 \
  data.label_enhancement=NONE \
  data.text_features_source=TXT \
  data.batch_size=32 \
  data.num_workers=4 \
  data.folds=[0]
```

Depois repetir para:

- `tasks=[predict]`
- `tasks=[eval]`
- `tasks=[fuse]`
- `tasks=[aggregate]`

### Passo 5. Rodar o script do dataset

Depois do smoke test, você pode usar o script integrado:

```bash
bash run/WOS-46985.sh 0 0
```

E depois:

```bash
bash run/WOS-46985.sh 0 9
```

## Observações importantes

### Ambiente Conda

Foi observado que o ambiente existente é `rag_fuse`, enquanto [run.sh](/home/flaviossf/work/benchmark-celso/RAG-Fuse/run.sh) tenta ativar `RAG-Fuse`.

Isso pode quebrar a execução direta de `run.sh` no seu ambiente atual.

Se isso acontecer, use:

```bash
conda run -n rag_fuse ...
```

ou ajuste manualmente o `conda activate` do script antes de usar.

### Notebook interrompido

A atualização do notebook foi interrompida antes de ser aplicada. O estado atual é:

- scripts criados: sim
- dataset local gerado: sim
- staging do HF gerado: sim
- notebook reescrito: não
- upload ao HF: não
- smoke test: não

## Resumo curto

Já está pronto:

- fonte oficial validada
- dataset bruto copiado
- artefatos do RAG-Fuse gerados
- config do dataset criada
- script `run/WOS-46985.sh` criado
- script de upload ao HF criado
- staging do HF pronto

Ainda falta:

- atualizar `RAGFuse_Datasets.ipynb`
- autenticar e publicar no Hugging Face
- rodar smoke test do fold `0`
- rodar benchmark completo `0..9`
