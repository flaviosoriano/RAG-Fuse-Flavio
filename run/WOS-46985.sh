# overrides
data=WOS-46985
model=RetrieverBERT

text_max_length=256
label_max_length=16
label_enhancement=NONE
text_features_source=TXT


#fit_batch_size=32
#infer_batch_size=64
#num_workers=4
#model_tag=NONE_RetrieverBERT

#mkdir -p resource/time
#export RAGFUSE_RUNTIME_HOME=${RAGFUSE_RUNTIME_HOME:-/tmp/ragfuse_home}
#export HOME=$RAGFUSE_RUNTIME_HOME
#export NLTK_DATA=${NLTK_DATA:-$HOME/nltk_data}
#export HF_HOME=${HF_HOME:-$HOME/hf_home}
#export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
#export MPLCONFIGDIR=${MPLCONFIGDIR:-$HOME/mpl_config}
#export IR_DATASETS_HOME=${IR_DATASETS_HOME:-$HOME/.ir_datasets}
#export IR_DATASETS_TMP=${IR_DATASETS_TMP:-$HOME/ir_datasets_tmp}
#export RETRIV_BASE_PATH=${RETRIV_BASE_PATH:-$HOME/.retriv}
#export NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT:-1}
#mkdir -p "$HOME" "$NLTK_DATA" "$HF_HOME" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$IR_DATASETS_HOME" "$IR_DATASETS_TMP" "$RETRIV_BASE_PATH"

## sparse_retrieve
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[sparse_retrieve] \
    model=BM25 \
    data=$data \
    data.text_features_source=$text_features_source \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/sparse_retrieve_${data}_${fold_idx}.tmr
done

## dense_retrieve fit
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[fit] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$model_tag \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=$fit_batch_size \
    data.num_workers=$num_workers \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fit_${model_tag}_${data}_${fold_idx}.tmr
done

## dense_retrieve predict
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[predict] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$model_tag \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=$infer_batch_size \
    data.num_workers=$num_workers \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/predict_${model_tag}_${data}_${fold_idx}.tmr
done

## dense_retrieve eval
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[eval] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$model_tag \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=$infer_batch_size \
    data.num_workers=$num_workers \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/eval_${model_tag}_${data}_${fold_idx}.tmr
done

## fuse
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[fuse] \
    model=$model \
    model.name=$model_tag \
    data=$data \
    data.text_features_source=$text_features_source \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fuse_${model_tag}_${data}_${fold_idx}.tmr
done

## aggregate
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[aggregate] \
    model=$model \
    model.name=$model_tag \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=$infer_batch_size \
    data.num_workers=$num_workers \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/aggregate_${model_tag}_${data}_${fold_idx}.tmr
done
