export PYTHONPATH=${PWD} 

# for FOLD_NUM in {0..4}
# do
#     CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
#         'experience.experiment_name=F10_AUDIO_FOLD${env:FOLD_NUM}_adamw_cosinelr_smclr_roadmap' \
#         experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
#         experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
#         experience.test_eval_freq=2 \
#         experience.save_model=10 \
#         experience.seed=42 \
#         experience.max_iter=30 \
#         'experience.split=${env:FOLD_NUM}' \
#         experience.kfold=10 \
#         experience.split_random_state=42 \
#         optimizer=adamw_cosinelr \
#         loss_optimizer=adamw \
#         model=simclr \
#         transform=img_transform \
#         dataset=audio_embeddings \
#         dataset.sampler.kwargs.batch_size=512 \
#         loss=roadmap
# done

CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD0_adamw_cosinelr_smclr_roadmap' \
    experience.split=0 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    dataset.kwargs.crop_size=81 \
    loss=roadmap


CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD1_adamw_cosinelr_smclr_roadmap' \
    experience.split=1 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    dataset.kwargs.crop_size=60 \
    loss=roadmap



CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD2_adamw_cosinelr_smclr_roadmap' \
    experience.split=2 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    loss=roadmap


CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD3_adamw_cosinelr_smclr_roadmap' \
    experience.split=3 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    loss=roadmap


CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD4_adamw_cosinelr_smclr_roadmap' \
    experience.split=4 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    loss=roadmap


CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD5_adamw_cosinelr_smclr_roadmap' \
    experience.split=5 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    loss=roadmap 


CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD6_adamw_cosinelr_smclr_roadmap' \
    experience.split=6 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    loss=roadmap 


CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD7_adamw_cosinelr_smclr_roadmap' \
    experience.split=7 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    loss=roadmap 


CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD8_adamw_cosinelr_smclr_roadmap' \
    experience.split=8 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    loss=roadmap 


CUDA_VISIBLE_DEVICES=0 python single_experiment_runner.py \
    'experience.experiment_name=F10_AUDIO_FOLD9_adamw_cosinelr_smclr_roadmap' \
    experience.split=9 \
    experience.kfold=10 \
    experience.log_dir=/home/and/projects/hacks/yandex_cup_2022/experiments/audio \
    experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features \
    experience.test_eval_freq=2 \
    experience.save_model=10 \
    experience.seed=42 \
    experience.max_iter=30 \
    experience.split_random_state=42 \
    optimizer=adamw_cosinelr \
    loss_optimizer=adamw \
    model=simclr \
    transform=img_transform \
    dataset=audio_embeddings \
    dataset.sampler.kwargs.batch_size=512 \
    loss=roadmap 
