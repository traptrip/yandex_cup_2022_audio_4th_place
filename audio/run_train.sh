PYTHONPATH=${PWD} python single_experiment_runner.py \
  'experience.experiment_name=AUDIO_modelopt_${optimizer[0].name}_lossopt_${loss_optimizer[0].name}_${dataset.sampler.kwargs.batch_size}' \
  'experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio' \
  'experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features' \
  experience.n_classes=18468 \
  experience.test_eval_freq=2 \
  experience.save_model=5 \
  experience.seed=42 \
  experience.max_iter=100 \
  'dataset.kwargs.csv_path=${experience.data_dir}/../train_meta_with_stages.tsv' \
  optimizer=adamw_cosinelr \
  loss_optimizer=adamw_cosinelr \
  model=simclr \
  dataset=audio_embeddings \
  dataset.sampler.kwargs.batch_size=1024 \
  loss=roadmap