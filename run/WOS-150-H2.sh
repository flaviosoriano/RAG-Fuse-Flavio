# overrides
data=WOS-150-H2
model=RetrieverBERT

text_max_length=256
label_max_length=256
label_enhancement=LLM
text_features_source=TXT
model_name=${label_enhancement}_${model}
sparse_model=BM25
dense_num_workers=${DENSE_NUM_WORKERS:-16}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# sparse_retrieve
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[sparse_retrieve] \
    model=$sparse_model \
    data=$data \
    data.text_features_source=$text_features_source \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/sparse_retrieve_${data}_${fold_idx}.tmr
done

# dense_retrieve fit
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[fit] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$model_name \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=96 \
    data.num_workers=$dense_num_workers \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fit_${model_name}_${data}_${fold_idx}.tmr
done

# dense_retrieve predict
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[predict] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$model_name \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=64 \
    data.num_workers=$dense_num_workers \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/predict_${model_name}_${data}_${fold_idx}.tmr
done

# dense_retrieve eval
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[eval] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=$model_name \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=64 \
    data.num_workers=$dense_num_workers \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/eval_${model_name}_${data}_${fold_idx}.tmr
done

# fuse
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[fuse] \
    model=$model \
    model.name=$model_name \
    data=$data \
    data.text_features_source=$text_features_source \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fuse_${model_name}_${data}_${fold_idx}.tmr
done

# aggregate
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[aggregate] \
    model=$model \
    model.name=$model_name \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=64 \
    data.num_workers=$dense_num_workers \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/aggregate_${model_name}_${data}_${fold_idx}.tmr
done
