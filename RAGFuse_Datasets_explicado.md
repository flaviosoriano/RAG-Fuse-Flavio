# RAGFuse_Datasets.ipynb explicado minuciosamente

## Objetivo do notebook

O notebook [RAGFuse_Datasets.ipynb](/home/flaviossf/work/benchmark-celso/RAG-Fuse/RAGFuse_Datasets.ipynb) e o ponto de preparacao dos datasets usados pelo projeto. Ele nao treina modelo, nao executa retrieval e nao faz fusao de rankings. O papel dele e transformar colecoes brutas em um contrato de artefatos serializados que o restante do pipeline consegue consumir.

Observado no codigo atual: a saida esperada por dataset e composta principalmente por:

- `resource/dataset/<DATASET>/samples.pkl`
- `resource/dataset/<DATASET>/fold_<k>/{train,val,test}.pkl`
- `resource/dataset/<DATASET>/relevance_map.pkl`
- `resource/dataset/<DATASET>/label_cls.pkl`
- `resource/dataset/<DATASET>/text_cls.pkl`

No caso de `WOS-46985`, o notebook tambem grava `metadata.json`.

Em termos de ideia metodologica, este notebook faz a ponte entre o dado cru e a formulacao de RAG-Fuse como problema de retrieval de labels:

- cada texto vira uma consulta;
- cada label vira um candidato relevante para esse texto;
- `relevance_map.pkl` define a relacao texto -> labels relevantes;
- `label_cls.pkl` e `text_cls.pkl` introduzem a separacao `head`/`tail`;
- os folds determinam a divisao experimental.

## Estrutura geral do notebook

O notebook tem 236 celulas e esta organizado em tres blocos principais:

1. `setup` e importacoes.
2. `helpers`, com funcoes de leitura, split, checkpoint e inspecao.
3. `Main`, com um pipeline repetido para varios datasets e dois casos especiais no final: `RCV1` e `WOS-46985`.

O padrao dominante do notebook e:

1. definir nome do dataset e nomes das labels;
2. baixar ou localizar os arquivos brutos;
3. montar `samples_df`;
4. classificar labels em `head` e `tail`;
5. derivar `text_cls`;
6. inspecionar estatisticas de comprimento;
7. montar folds;
8. gerar `relevance_map`;
9. salvar tudo em disco.

## Parte 1: setup

### Celula 0

Titulo visual do bloco: `# setup`.

Ela nao executa logica, apenas separa semanticamente o notebook.

### Celula 1

Importa todas as dependencias usadas ao longo do notebook:

- `pandas`, `numpy`, `pickle`, `json`, `Path`, `os`, `subprocess`
- `Counter`
- `random`
- `StratifiedKFold`, `KFold`
- `fetch_rcv1`
- `seaborn`
- `tqdm`
- `networkx`
- `CountVectorizer`

O que isso revela sobre o notebook:

- ele mistura preparacao tabular (`pandas`) com serializacao (`pickle`);
- gera folds tanto simples quanto estratificados;
- tem um caso especial com dataset esparso (`RCV1`);
- inclui bibliotecas que nao sao centrais no fluxo final, como `networkx` e `CountVectorizer`.

Observacao importante: nem tudo que e importado e realmente usado no fluxo principal. Isso sugere que o notebook acumulou trechos exploratorios ao longo do tempo.

## Parte 2: helpers

### Celulas 2 a 4: leitura basica e download

#### Celula 2

Titulo visual do bloco: `## helpers`.

#### Celula 3

Define as funcoes basicas de leitura e montagem inicial dos dados.

`load_samples(dataset)`

- Abre `resource/dataset/<dataset>/samples.pkl`.
- Carrega o artefato ja processado.
- E um helper de consumo de saida, nao de criacao.

`get_scores(scores_path)`

- Le um arquivo texto contendo um score ou classe por linha.
- Converte cada linha para inteiro.
- Retorna um `DataFrame` com coluna `cls`.

`get_texts(texts_path)`

- Le um arquivo binario linha a linha.
- Decodifica em UTF-8.
- Remove quebras de linha e `\r`.
- Retorna um `DataFrame` com coluna `text`.

`get_splits(splits_path)`

- Desserializa um arquivo pickle contendo uma tabela de splits.
- Essa funcao reaparece depois, redefinida de forma identica na celula 6.

`get_samples(texts_path, scores_path)`

- Junta os textos e os scores.
- Cria `idx` como o indice da linha.
- Refatoriza `cls` com `pd.factorize(..., sort=True)`.
- Retorna somente `idx`, `text` e `cls`.

Impacto pratico:

- o notebook parte de uma representacao simples: um texto por linha e uma classe por linha;
- o formato bruto e imediatamente convertido para um dataframe padrao;
- `cls` e remapeado para ids inteiros consecutivos.

#### Celula 4

`download_dataset(dataset, dataset_dir, base_url=...)`

- Cria o diretorio alvo.
- Chama `wget` recursivo contra `http://150.164.2.44/datasets/<dataset>/`.
- Salva os arquivos baixados em `resource/dataset/raw_dataset/<DATASET>/`.

O que a celula faz conceitualmente:

- automatiza a etapa de aquisicao do dado bruto.

Risco observado:

- a reproducibilidade depende de um servidor HTTP externo;
- o notebook assume disponibilidade de `wget`;
- nao ha validacao de hash, versao ou integridade dos arquivos.

### Celulas 5 a 10: geracao e leitura de folds

#### Celula 5

Titulo visual do subbloco: `### folds`.

#### Celula 6

Redefine `get_splits(splits_path)` exatamente para ler um pickle com splits.

Observado no codigo atual:

- essa redefinicao nao muda comportamento;
- ela parece redundante.

#### Celula 7

Concentra a logica geral de particionamento.

`train_val_split(ids, split_indice=.9)`

- embaralha a lista recebida in-place com `random.shuffle`;
- quebra a lista em treino e validacao no ponto `90% / 10%`.

`k_fold_split(ids, n_splits=5, ...)`

- usa `KFold`;
- para cada fold, separa indices de treino+validacao e teste;
- reaproveita `train_val_split` para separar treino de validacao;
- retorna uma lista de dicionarios com a estrutura:
  `{"fold_idx": ..., "splits": {"train": [...], "val": [...], "test": [...]}}`

`sk_fold_split(ids, cls, n_splits=5, ...)`

- mesmo formato de saida;
- usa `StratifiedKFold`;
- preserva aproximadamente a distribuicao da classe `cls` entre folds.

`show_folds(folds, samples_to_show=10)`

- imprime um resumo de cada fold;
- serve para inspecao manual, nao para persistencia.

Ponto metodologico:

- o notebook trata a qualidade da divisao experimental como parte da preparacao do dataset, nao como detalhe do treino.

#### Celula 8

Essa e uma das partes mais importantes para os casos multilabel e esparsos.

`csr_row_to_text(csr_matrix, row_idx, max_terms=256)`

- pega uma linha de uma matriz CSR;
- extrai indices de termos ativos;
- transforma esses ids em tokens artificiais como `w17`, `w204`, `w999`;
- limita o pseudo-texto a `max_terms`.

Uso conceitual:

- permite converter a representacao esparsa de `RCV1` em um texto sintetico que ainda pode ser consumido pelo restante do pipeline.

`greedy_multilabel_fold_indices(...)`

- implementa uma estratificacao multilabel aproximada;
- calcula frequencias totais por label;
- estima o alvo por fold;
- ordena amostras priorizando maior cardinalidade de labels e labels raras;
- aloca cada amostra no fold que mais reduz desequilibrio.

`ml_k_fold_split(...)`

- usa o algoritmo guloso acima para decidir conjuntos de teste por fold;
- separa o restante em treino e validacao;
- devolve a mesma estrutura de `folds` usada no restante do notebook.

Impacto experimental:

- esse bloco existe para evitar folds ruins em problemas multilabel;
- sem ele, `RCV1` teria maior risco de labels raras desaparecerem em alguns folds.

#### Celula 9

Executa um microteste de `k_fold_split` com `ids = [0, 1, 2, 3]`.

Serve apenas como sanity check local.

#### Celula 10

`load_K_fold_split(splits_df)`

- recebe um dataframe com colunas como `fold_id`, `train_idxs`, `val_idxs`, `test_idxs`;
- converte para o formato de lista de folds usado pelo notebook.

Essa funcao e a ponte entre splits precomputados baixados do dataset bruto e o formato interno do projeto.

### Celulas 11 a 18: checkpoints

Esse bloco transforma estruturas em memoria em artefatos persistidos.

#### Celula 12

`checkpoint_samples(samples_df, dataset_dir)`

- reduz o dataframe para as colunas canonicas:
  `idx`, `text_idx`, `text`, `labels_ids`, `labels`
- grava `samples.pkl`.

Esse arquivo e o contrato mais importante do notebook.

#### Celula 13

`checkpoint_folds(folds, dataset_dir)`

- cria `fold_<k>/`;
- grava `train.pkl`, `val.pkl` e `test.pkl`.

#### Celula 14

`checkpoint_relevance_map(relevance_map, dataset_dir)`

- grava `relevance_map.pkl`.

#### Celula 15

`checkpoint_label_cls(label_cls, dataset_dir)`

- grava `label_cls.pkl`.

#### Celula 16

`checkpoint_labels_propensities(...)`

- existe, mas nao aparece sendo usada nos blocos principais do notebook.

#### Celula 17

`checkpoint_text_cls(text_cls, dataset_dir)`

- grava `text_cls.pkl`.

#### Celula 18

`checkpoint_labels_map(labels_map, dataset_dir)`

- helper presente, mas nao usado nos pipelines mostrados.

Leitura operacional:

- o notebook salva apenas o que o pipeline precisa de forma explicita;
- `labels_propensities` e `labels_map` parecem preparacao para fluxos auxiliares ou evolucoes futuras.

### Celulas 19 a 24: distribuicao e estatisticas

#### Celula 20

Carrega o dataset `titanic` do `seaborn` e plota uma contagem por classe.

Observado no codigo atual:

- isso e um exemplo isolado;
- nao participa da preparacao de nenhum dataset do projeto.

#### Celula 21

`plot_label_distribution(samples_df, column)`

- percorre as labels de cada amostra;
- achata tudo numa lista;
- desenha um `countplot`.

Uso:

- inspecionar desbalanceamento de labels.

#### Celula 23

`get_max_num_tokens(labels)`

- calcula o maior numero de tokens entre os nomes das labels.

`len_stats(df)`

- cria uma copia do dataframe;
- calcula o comprimento do texto em tokens separados por espaco;
- mostra quantis do comprimento do texto;
- calcula o comprimento conjunto das labels de cada amostra;
- mostra quantis do comprimento das labels.

Utilidade metodologica:

- ajuda a entender se o dataset tem textos longos ou curtos;
- ajuda a estimar quao informativos sao os nomes das labels;
- ajuda a avaliar lacunas lexicais entre texto e label.

#### Celula 24

`inspect_stat(dataset)`

- abre `samples_with_keywords.pkl` e `label_cls.pkl`;
- conta, por amostra, quantas labels `tail` e `head` existem.

Observado no codigo atual:

- a funcao parece incompleta;
- no trecho visivel ela nao retorna nada nem exibe resumo final.

Isso sugere um helper exploratorio ou parcialmente abandonado.

### Celulas 25 a 28: relevancia e classes derivadas

#### Celula 26

`get_relevance_map(df)`

- constroi um mapa de relevancia entre `qs1_idx` e `qs2_idx`;
- considera relevante apenas linhas com `cls == 1`.

Importante:

- esse helper nao e o usado pelos blocos principais de datasets;
- ele parece vir de um cenario de pares consulta-documento ou consulta-consulta.

#### Celula 28

`get_text_cls(label_ids, label_cls)`

- agrega as classes associadas a cada label de uma amostra;
- retorna a uniao sem duplicatas.

Na pratica:

- se um texto tem labels `head` e `tail`, esse texto pode receber ambas as tags em `text_cls`.

## Parte 3: bloco principal por dataset

### O template dominante

Da celula 29 em diante, o notebook repete quase sempre o mesmo molde:

1. declara nomes das labels;
2. baixa ou localiza o dataset bruto;
3. gera `samples_df`;
4. calcula `label_cls`;
5. calcula `text_cls`;
6. mede comprimentos;
7. carrega ou gera folds;
8. monta `relevance_map`;
9. salva artefatos.

Esse padrao aparece em:

- `ACM` nas celulas 30 a 47
- `REUTERS` nas celulas 48 a 65
- `OHSUMED` nas celulas 66 a 83
- `WOS` nas celulas 84 a 101
- `DBLP` nas celulas 102 a 119
- `20NG` nas celulas 120 a 137
- `BOOKS` nas celulas 138 a 155
- `AGNEWS` nas celulas 156 a 173
- `SST2` nas celulas 174 a 192
- `TREC` nas celulas 193 a 211

### Como ler um bloco padrao usando ACM como exemplo

#### Celulas 30 e 31

Abrem a secao `## ACM` e definem:

- `dataset = "ACM"`
- `dataset_raw_name = "acm"`
- `labels_names = [...]`

Aqui o notebook fixa o mapeamento semantico entre ids inteiros e nomes humanos das labels.

#### Celula 32

Baixa o bruto de `ACM` para `resource/dataset/raw_dataset/ACM/`.

#### Celulas 33 e 34: samples

`samples_df = get_samples(...)`

Em seguida o dataframe e enriquecido com:

- `labels = [[labels_names[x]]]`
- `labels_ids = [[x]]`
- `text_idx = samples_df["text"].astype("category").cat.codes`

No fim, o dataframe fica nas cinco colunas canonicas:

- `idx`
- `text_idx`
- `text`
- `labels_ids`
- `labels`

Interpretacao:

- mesmo em datasets single-label, o notebook normaliza tudo para listas, mantendo compatibilidade com o formato multilabel;
- `text_idx` agrupa textos identicos sob o mesmo codigo categorico.

#### Celulas 35 e 36: labels cls

Conta a frequencia de cada label com `Counter`.

Depois define `cutoff = round(0.2 * len(labels_counter))` e separa labels em:

- `["all", "head"]`
- `["all", "tail"]`

Detalhe importante observado no codigo atual:

- a regra usa `if len(head) <= cutoff`, nao `< cutoff`;
- isso tende a colocar `cutoff + 1` labels em `head`, nao exatamente `20%`.

Conceitualmente:

- o notebook nao aprende `head`/`tail` por limiar absoluto de frequencia;
- ele pega as labels mais frequentes em ordem e marca um bloco inicial como `head`.

#### Celulas 37 e 38: text cls

Para cada `text_idx`, agrega as classes derivadas de suas labels.

Resultado:

- `text_cls[text_idx] = ["all", "head"]`, ou
- `["all", "tail"]`, ou
- `["all", "head", "tail"]` em cenarios multilabel.

#### Celulas 39 e 40: len stats

Executam a inspecao de quantis de comprimento do texto e das labels.

Isso nao altera artefatos; serve para diagnostico.

#### Celulas 41 a 43: folds

Leem um arquivo precomputado como `split_10_with_val.pkl`, convertem com `load_K_fold_split` e imprimem exemplos.

Leitura operacional:

- para a maioria dos datasets classificados de forma convencional, os folds nao sao gerados do zero aqui;
- o notebook importa folds ja existentes do dataset bruto.

#### Celulas 44 e 45: relevance map

`relevance_map = pd.Series(samples_df["labels_ids"].values, index=samples_df["text_idx"]).to_dict()`

Isso produz um dicionario:

- chave: `text_idx`
- valor: lista de `labels_ids` relevantes para esse texto

Ponto importante:

- em datasets single-label, isso representa exatamente uma label por texto;
- se houver textos duplicados com o mesmo `text_idx`, `to_dict()` preserva apenas a ultima ocorrencia daquela chave.

Essa decisao parece segura apenas quando duplicatas nao introduzem conflito de labels.

#### Celulas 46 e 47: checkpoint

Criam o diretorio `resource/dataset/ACM/` e gravam:

- `samples.pkl`
- folds
- `relevance_map.pkl`
- `label_cls.pkl`
- `text_cls.pkl`

Esse mesmo desenho se repete nos outros datasets "padrao".

## Diferencas entre os blocos padrao

### REUTERS, OHSUMED, WOS, DBLP, 20NG e BOOKS

Esses blocos repetem praticamente o molde de `ACM` sem alteracao estrutural.

O que muda:

- `dataset`
- `dataset_raw_name`
- `labels_names`
- o arquivo de split bruto correspondente

O contrato final salvo continua o mesmo.

### AGNEWS

O bloco de `AGNEWS` fica nas celulas 156 a 173.

Diferenca principal observada:

- usa `split_5_with_val.pkl` na celula 168, nao `split_10_with_val.pkl`.

Implicacao experimental:

- comparacoes com outros datasets precisam lembrar que aqui a validacao cruzada foi preparada com 5 folds, nao 10.

### SST2

O bloco de `SST2` fica nas celulas 174 a 192.

Ha uma celula extra de diagnostico, a 179:

- `print(len(samples_df), samples_df.shape[0], samples_df["text_idx"].nunique())`

O objetivo pratico dessa impressao e comparar:

- numero total de linhas;
- numero de exemplos no dataframe;
- numero de textos unicos depois do agrupamento em `text_idx`.

Isso indica preocupacao com textos repetidos.

### TREC

O bloco de `TREC` fica nas celulas 193 a 211.

Tambem inclui uma celula extra de diagnostico, a 198, com a mesma checagem de unicidade de `text_idx`.

Leitura metodologica:

- em datasets curtos ou com perguntas muito semelhantes, a colisao de `text_idx` pode afetar `relevance_map` e `text_cls`.

## Bloco especial 1: RCV1

O bloco `RCV1` ocupa as celulas 212 a 219 e e conceitualmente diferente do restante.

### Celula 212

Titulo explicativo: `RCV1-v1 from sklearn.datasets.fetch_rcv1 (multilabel)`.

### Celula 213

Inicializa:

- `dataset = "RCV1"`
- `dataset_raw_name = "rcv1-v2"`
- `rcv1 = fetch_rcv1(subset="all", download_if_missing=True)`
- `labels_names = rcv1.target_names.tolist()`

O notebook aqui nao usa o servidor HTTP dos outros datasets. Ele usa o carregador do `scikit-learn`.

### Celula 214: construcao de samples

Essa e a transformacao central do bloco.

Passos:

1. define `max_terms_per_text = 256`;
2. converte `target` e `data` para CSR;
3. escolhe todos os indices ou um subconjunto se `max_samples` for definido;
4. para cada amostra:
   - extrai a lista de labels ativas;
   - converte a linha esparsa de texto para um pseudo-texto via `csr_row_to_text`;
5. constroi `samples_df` com `idx`, `text`, `labels_ids`;
6. deriva `labels` com nomes humanos;
7. gera `text_idx`.

O que esta acontecendo conceitualmente:

- `RCV1` nao chega como texto natural pronto;
- o notebook fabrica um texto artificial a partir de ids de termos ativos;
- isso preserva compatibilidade com o restante do pipeline de retrieval.

Risco metodologico:

- esse pseudo-texto carrega apenas ids lexicais artificiais, nao a superficie textual original;
- comparacoes com datasets textuais naturais precisam considerar essa diferenca.

### Celula 215: labels cls

Repete a mesma logica `head`/`tail` dos datasets padrao.

### Celula 216: relevance map

Aqui o notebook muda a implementacao para evitar perda de labels.

Passos:

- cria `relevance_map` como dicionario de conjuntos;
- agrega labels por `text_idx`;
- ao final, converte cada conjunto para lista ordenada.

Diferenca importante em relacao aos datasets padrao:

- nao usa `Series(...).to_dict()`;
- evita sobrescrita silenciosa quando multiplas linhas compartilham o mesmo `text_idx`.

### Celula 217: folds multilabel

Usa `ml_k_fold_split(...)` com:

- ids das amostras;
- labels multilabel por amostra;
- numero total de labels;
- 10 folds.

Esse e um ajuste metodologico importante para o caso multilabel.

### Celula 218: text cls

Agrupa classes por `text_idx` usando conjuntos, depois ordena.

Novamente, o foco e evitar perda de informacao quando textos iguais aparecem mais de uma vez.

### Celula 219: checkpoint

Salva os mesmos artefatos centrais do restante:

- `samples.pkl`
- folds
- `relevance_map.pkl`
- `label_cls.pkl`
- `text_cls.pkl`

## Bloco especial 2: WOS-46985

O bloco `WOS-46985` ocupa as celulas 220 a 235 e documenta uma variante hierarquica especifica.

### Celula 220

Explica a semantica dos arquivos:

- `Y`: label plana canonica com 134 classes
- `YL1`: label pai com 7 classes
- `YL2`: codigo de nivel 2 reutilizado em pais diferentes

Ponto central observado:

- `YL2` sozinho nao identifica univocamente as 134 classes;
- por isso o notebook escolhe `Y` como alvo plano canonico.

### Celula 221

Define:

- nomes do dataset e variante;
- contagens esperadas de amostras e labels;
- diretorio fonte local `resource/dataset/WOS-46985`;
- conjunto de arquivos obrigatorios.

Tambem faz validacoes:

- verifica existencia do diretorio;
- verifica se `X.txt`, `Y.txt`, `YL1.txt` e `YL2.txt` estao presentes.

### Celula 223: samples

Passos:

1. le `X.txt`, `Y.txt`, `YL1.txt`, `YL2.txt`;
2. checa consistencia do numero de linhas;
3. checa quantidade esperada de amostras;
4. calcula:
   - numero de labels unicas em `Y`
   - numero de labels unicas em `YL1`
   - numero de labels unicas em `YL2`
   - numero de pares unicos `(YL1, YL2)`
   - numero de triplas unicas `(Y, YL1, YL2)`
5. avisa se `YL2` estiver sendo reutilizado entre pais;
6. fatoriza `Y` para obter `labels_ids_factorized`;
7. constroi `samples_df` usando `Y` como label final.

Interpretacao metodologica:

- o notebook esta explicitamente corrigindo uma ambiguidade do dataset hierarquico;
- ele nao assume que `YL2` seja uma label folha suficiente;
- ele privilegia `Y` como verdade de referencia para o alvo flat.

### Celulas 224 a 233

Repetem o mesmo esqueleto dos datasets single-label:

- contagem de frequencia de labels e divisao `head`/`tail`
- construcao de `text_cls`
- estatisticas de comprimento
- folds estratificados com `sk_fold_split`
- `relevance_map` simples por `text_idx`

### Celula 235: checkpoint e metadata

Salva:

- `samples.pkl`
- folds
- `relevance_map.pkl`
- `label_cls.pkl`
- `text_cls.pkl`

E tambem grava `metadata.json` com:

- nome do dataset
- variante usada
- contagens de amostras e labels
- semantica de `Y`, `YL1`, `YL2`
- caminhos dos arquivos fonte

Essa e a parte mais documental do notebook. Ela registra explicitamente a interpretacao correta da variante local.

## O que cada artefato significa no pipeline

### `samples.pkl`

Lista de amostras padronizadas. Cada item contem:

- `idx`: id da amostra
- `text_idx`: id do texto apos agrupamento por igualdade textual
- `text`: conteudo textual ou pseudo-textual
- `labels_ids`: ids inteiros das labels relevantes
- `labels`: nomes das labels relevantes

### `fold_<k>/{train,val,test}.pkl`

Indices de amostras usados em cada particao experimental.

### `relevance_map.pkl`

Mapa `text_idx -> labels_ids relevantes`.

No contexto de RAG-Fuse, esse artefato aproxima a nocao de "documentos relevantes" por consulta, so que aqui os documentos relevantes sao labels.

### `label_cls.pkl`

Mapa `label_id -> classes auxiliares`, tipicamente:

- `["all", "head"]`
- `["all", "tail"]`

Ele injeta a logica de desbalanceamento no dataset preparado.

### `text_cls.pkl`

Mapa `text_idx -> classes auxiliares derivadas das labels do texto`.

Serve para rotular a consulta segundo a composicao `head`/`tail` das suas labels relevantes.

### `metadata.json` em `WOS-46985`

Registra a semantica da variante hierarquica e reduz ambiguidade interpretativa.

## Observacoes importantes sobre o notebook atual

### 1. O notebook mistura preparo canonico com exploracao

Exemplos:

- celula do `titanic`
- helpers nao usados no fluxo principal
- `inspect_stat` aparentemente incompleto

### 2. `head` e `tail` sao definidos por ranking de frequencia, nao por corte estatistico sofisticado

Isso importa porque:

- a divisao depende do numero de labels presentes;
- a regra atual produz aproximadamente os 20% mais frequentes como `head`;
- devido ao uso de `<=`, o tamanho de `head` pode ser `cutoff + 1`.

### 3. `text_idx` colapsa textos identicos

Isso e intencional no contrato atual, mas pode ter implicacoes:

- `relevance_map` e `text_cls` passam a ser indexados pelo texto, nao pela linha original;
- em casos com textos repetidos e labels diferentes, ha risco de sobrescrita ou agregacao inesperada;
- `RCV1` trata isso com mais cuidado do que varios datasets padrao.

### 4. O notebook padroniza single-label como multilabel de cardinalidade 1

Isso e importante para o restante do repositorio:

- o pipeline pode operar com um contrato unico;
- a diferenca entre single-label e multilabel fica principalmente na cardinalidade de `labels_ids`.

## Resumo executivo

Se voce quiser pensar no notebook de forma compacta, ele faz quatro coisas:

1. le dados crus e os converte para um dataframe canonico;
2. define como os experimentos serao divididos em folds;
3. explicita a relacao texto -> labels relevantes e a taxonomia `head`/`tail`;
4. grava tudo no formato que o restante do RAG-Fuse espera.

Os dois desvios importantes do padrao sao:

- `RCV1`, porque exige conversao de matriz esparsa para pseudo-texto e folds multilabel;
- `WOS-46985`, porque exige desambiguar a hierarquia e fixar `Y` como label plana canonica.

## Arquivos e celulas mais importantes para revisar primeiro

Se o objetivo for entender rapidamente o notebook, comece por esta ordem:

1. celulas 12 a 18, para ver quais artefatos ele grava;
2. celula 34, para entender o formato canonico de `samples_df`;
3. celula 36, para ver como `head` e `tail` sao definidos;
4. celula 45, para entender o `relevance_map` padrao;
5. celulas 214 a 218, para entender o caso especial de `RCV1`;
6. celula 223 e 235, para entender a interpretacao de `WOS-46985`.

## Conclusao

O notebook e menos um "dataset explorer" e mais um compilador de datasets para o contrato operacional do projeto. Ele transforma fontes heterogeneas em uma representacao comum que preserva:

- texto da consulta;
- labels relevantes;
- divisao experimental;
- informacao auxiliar de desbalanceamento.

Essa uniformizacao e justamente o que permite ao restante do RAG-Fuse tratar o problema como retrieval de labels em vez de classificacao tradicional direta.
