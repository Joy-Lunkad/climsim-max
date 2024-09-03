# v4-8
git clone https://github.com/Joy-Lunkad/climsim-max.git --branch re_sub_2
cd climsim-max
bash setup.sh MODE=stable JAX_VERSION=0.4.28

echo 'export PATH=$PATH:/home/joylunkad/.local/bin' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages' >> ~/.bashrc
source ~/.bashrc

gsutil cp gs://us2-climsim/sample_id_to_sample_weights.csv ./

kaggle -v
gsutil cp gs://us2-climsim/kaggle.json ./
cp kaggle.json ~/.kaggle/

rm -f ~/.kaggle/kaggle.json && gsutil cp gs://us2-climsim/lunkad_tv_kaggle.json ./ && cp lunkad_tv_kaggle.json ~/.kaggle/kaggle.json


screen -S train

git pull && python3 MaxText/train.py MaxText/configs/base.yml num_epochs=201 use_full_low_res_data=True full_low_res_version=7.0.0 only_mlp=True base_num_decoder_layers=4 base_mlp_dim=8192 per_device_batch_size=4096 learning_rate=1e-2 cosine_learning_rate_final_fraction=1e-5 run_name=test_mlp_with_7


## v4-64

## Set zone and project
gcloud config set project ai-memory
gcloud config set compute/zone us-central2-b
ssh-keygen -f ~/.ssh/google_compute_engine

git clone https://github.com/Joy-Lunkad/climsim-max.git --branch re_sub_2
cd climsim-max
gsutil cp gs://us2-climsim/sample_id_to_sample_weights.csv ./

echo 'export PATH=$PATH:/home/joylunkad/.local/bin' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages' >> ~/.bashrc

export PATH=$PATH:/home/joylunkad/.local/bin
export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages


python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="source ~/.bashrc && bash setup.sh MODE=stable JAX_VERSION=0.4.28"

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="source ~/.bashrc && /home/joylunkad/.local/bin/kaggle -v"

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="gsutil cp gs://us2-climsim/kaggle.json ./ && cp kaggle.json ~/.kaggle/"

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="rm -f ~/.kaggle/kaggle.json && gsutil cp gs://us2-climsim/lunkad_tv_kaggle.json ./ && cp lunkad_tv_kaggle.json ~/.kaggle/kaggle.json"

git pull && python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml per_device_batch_size=64 num_epochs=20 base_num_decoder_layers=12 base_emb_dim=512 base_mlp_dim=2048 base_num_query_heads=4 base_num_kv_heads=4  use_full_low_res_data=True mix_high_res_ratio=0.375 learning_rate=1e-3 ici_fsdp_parallelism=4 run_name=psuedo_moe_mixed_ds_nofsmlp"

git pull && python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml per_device_batch_size=256 num_epochs=20 base_emb_dim=128 base_mlp_dim=512 base_num_query_heads=4 base_num_kv_heads=4 use_feature_specific_mlp=True use_full_low_res_data=True mix_high_res_ratio=0.25 learning_rate=1e-3 ici_fsdp_parallelism=4 load_parameters_path=gs://us2-climsim/maxtext/psuedo_moe_1/checkpoints/12199/items run_name=psuedo_moe_mixed_ds"


git pull && python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml per_device_batch_size=256 num_epochs=40 base_emb_dim=128 base_mlp_dim=512 base_num_query_heads=4 base_num_kv_heads=4 use_feature_specific_mlp=True use_full_low_res_data=True learning_rate=5e-4 ici_fsdp_parallelism=4 load_parameters_path=gs://us2-climsim/maxtext/psuedo_moe_1/checkpoints/12199/items gradient_clipping_threshold=0.01 cosine_learning_rate_final_fraction=0.01 run_name=psuedo_moe_2"

git pull && python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml only_mlp=True base_num_decoder_layers=8 base_mlp_dim=8192 per_device_batch_size=1024 learning_rate=1e-3 cosine_learning_rate_final_fraction=1e-5 num_epochs=10 adam_weight_decay=0.0 use_full_low_res_data=True load_parameters_path=gs://us2-climsim/maxtext/pretrain_only_mlp/checkpoints/41480/items  run_name=finetune_pretrained_mlp"

git pull && python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml only_mlp=True base_num_decoder_layers=8 base_mlp_dim=8192 per_device_batch_size=2048 learning_rate=1e-2 cosine_learning_rate_final_fraction=1e-5 num_epochs=5 adam_weight_decay=0.0 use_high_res_data=True run_name=pretrain_only_mlp "

git pull && python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml num_epochs=1 use_full_low_res_data=True full_low_res_version=2.0.0 enable_checkpointing=False run_name=test_2_again"

git pull && python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml num_epochs=20 use_full_low_res_data=True full_low_res_version=6.0.0  base_emb_dim=1024 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=4096 base_num_decoder_layers=12 per_device_batch_size=32.0 learning_rate=1e-4 run_name=216M_20ep load_parameters_path=gs://us2-climsim/maxtext/216M_10ep/checkpoints/97649/items"

git pull && python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml num_epochs=10 use_full_low_res_data=True full_low_res_version=6.0.0  base_emb_dim=1024 base_num_query_heads=8 base_num_kv_heads=8 base_mlp_dim=4096 base_num_decoder_layers=12 per_device_batch_size=32.0 learning_rate=1e-4 run_name=100M_10ep"


python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml num_epochs=20 use_full_low_res_data=True use_value_masks=True full_low_res_version=2.0.0 use_submission_trick=True run_name=test_sw_base_2 load_parameters_path=gs://us2-climsim/maxtext/test_sw_base/checkpoints/976/items"

screen -S train

python3 multihost_runner.py --TPU_PREFIX=us-node-3 --COMMAND="source ~/.bashrc && python3 MaxText/train.py MaxText/configs/base.yml run_name=re_recreate_sub_test_32 steps=20000"

python3 multihost_runner.py --TPU_PREFIX=us-node-1 --COMMAND="source ~/.bashrc && git pull && python3 MaxText/train.py MaxText/configs/recreate_sub_test_32.yml run_name=re_main_recreate_sub_test_32 steps=20000"

python3 multihost_runner.py --TPU_PREFIX=us-node-1 --COMMAND="source ~/.bashrc && git pull && python3 MaxText/train.py MaxText/configs/recreate_sub_test_32_changes.yml run_name=value_mask_feature_mlp_recreate_sub_test_32 steps=20000 use_feature_specific_mlp=True use_value_masks=True ici_data_parallelism=1 ici_fsdp_parallelism=-1 remat_policy='full' base_mlp_dim=512"

python3 multihost_runner.py --TPU_PREFIX=us-node-2 --COMMAND="source ~/.bashrc && git pull && python3 MaxText/train.py MaxText/configs/recreate_sub_test_32_changes.yml run_name=feature_mlp_recreate_sub_test_32 steps=20000 use_feature_specific_mlp=True ici_data_parallelism=1 ici_fsdp_parallelism=-1 remat_policy='full' base_mlp_dim=512"

python3 multihost_runner.py --TPU_PREFIX=us-node-4 --COMMAND="source ~/.bashrc && git pull && python3 MaxText/train.py MaxText/configs/recreate_sub_test_32_changes.yml run_name=value_masks_recreate_sub_test_32 steps=20000 use_value_masks=True"



python3 multihost_runner.py --TPU_PREFIX=us-node-3 --COMMAND="source ~/.bashrc && git pull && python3 MaxText/train.py MaxText/configs/v4/25m.yml run_name=25M_30_epochs_test_1 enable_checkpointing=False"


python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="gsutil cp gs://us2-climsim/kaggle.json ./ && cp kaggle.json ~/.kaggle/"
python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="git pull && python3 MaxText/train.py MaxText/configs/v4/25m.yml run_name=25M_30_epochs"
python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="git pull && python3 MaxText/train.py MaxText/configs/base.yml run_name=large_test"
python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="git pull && python3 MaxText/train.py MaxText/configs/base.yml run_name=25M_in_large_test base_emb_dim=512 base_num_query_heads=4 base_num_kv_heads=4 base_mlp_dim=2048 base_num_decoder_layers=12 num_epochs=3 warmup_steps_fraction=0.1 steps=-1 learning_rate=5.e-4 submit_to_kaggle=False"


gcloud alpha compute tpus queued-resources delete 25M_30_epochs --force --async

# Using multihost_job.py

BUCKET_NAME=gs://us2-climsim

python3 multihost_job.py --TPU_TYPE=v4-64 --NUM_SLICES=1 --RUN_NAME=25M_30_epochs --BUCKET_NAME=gs://us2-climsim --ENABLE_AUTOCHECKPOINT=True --CQR_EXTRA_ARGS="--best-effort" --COMMAND="bash setup.sh && git pull && python3 MaxText/train.py MaxText/configs/v4/25m.yml run_name=25M_30_epochs"

gcloud alpha compute tpus queued-resources create qr1-32 --node-id us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-32 --runtime-version tpu-ubuntu2204-base

gcloud alpha compute tpus queued-resources create qr1-64 --node-id us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-64 --runtime-version tpu-ubuntu2204-base

gcloud alpha compute tpus queued-resources create qr1-8 --node-id us-node-1 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr2-8 --node-id us-node-2 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr3-8 --node-id us-node-3 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr4-8 --node-id us-node-4 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr5-8 --node-id us-node-5 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr6-8 --node-id us-node-6 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr7-8 --node-id us-node-7 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr8-8 --node-id us-node-8 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base

gcloud config set project ai-memory
gcloud config set compute/zone us-central2-b
gcloud alpha compute tpus queued-resources list
gcloud alpha compute tpus queued-resources delete qr1-64 --force --async
gcloud alpha compute tpus queued-resources create qr1-64 --node-id us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-64 --runtime-version tpu-ubuntu2204-base


gcloud alpha compute tpus queued-resources delete qr1-8 --force --async
gcloud alpha compute tpus queued-resources delete qr2-8 --force --async
gcloud alpha compute tpus queued-resources delete qr3-8 --force --async
gcloud alpha compute tpus queued-resources delete qr4-8 --force --async
gcloud alpha compute tpus queued-resources delete qr5-8 --force --async
gcloud alpha compute tpus queued-resources delete qr6-8 --force --async
gcloud alpha compute tpus queued-resources delete qr7-8 --force --async
gcloud alpha compute tpus queued-resources delete qr8-8 --force --async

gcloud alpha compute tpus queued-resources delete pre-qr1-8 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr2-8 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr3-8 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr4-8 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr5-8 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr6-8 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr7-8 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr8-8 --force --async

gcloud alpha compute tpus queued-resources delete qr1-32 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr-3 --force --async
gcloud alpha compute tpus queued-resources delete pre-qr1 --force --async




gcloud alpha compute tpus queued-resources create qr1-16 --node-id us-node-1 --project ai-memory --zone us-central2-b --accelerator-type v4-16 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr2-16 --node-id us-node-2 --project ai-memory --zone us-central2-b --accelerator-type v4-16 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr3-16 --node-id us-node-3 --project ai-memory --zone us-central2-b --accelerator-type v4-16 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr4-16 --node-id us-node-4 --project ai-memory --zone us-central2-b --accelerator-type v4-16 --runtime-version tpu-ubuntu2204-base

gcloud alpha compute tpus queued-resources delete qr1-16 --force --async
gcloud alpha compute tpus queued-resources delete qr2-16 --force --async
gcloud alpha compute tpus queued-resources delete qr3-16 --force --async
gcloud alpha compute tpus queued-resources delete qr4-16 --force --async



gcloud alpha compute tpus queued-resources create qr1-32 --node-id us-node-1 --project ai-memory --zone us-central2-b --accelerator-type v4-32 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create qr2-32 --node-id us-node-2 --project ai-memory --zone us-central2-b --accelerator-type v4-32 --runtime-version tpu-ubuntu2204-base
gcloud alpha compute tpus queued-resources create pre-qr1-32 --node-id pre-us-node-1 --project ai-memory --zone us-central2-b --accelerator-type v4-32 --runtime-version tpu-ubuntu2204-base --best-effort
gcloud alpha compute tpus queued-resources create pre-qr2-32 --node-id pre-us-node-2 --project ai-memory --zone us-central2-b --accelerator-type v4-32 --runtime-version tpu-ubuntu2204-base --best-effort



gcloud alpha compute tpus queued-resources create pre-qr1-8 --node-id pre-us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base --best-effort
gcloud alpha compute tpus queued-resources create pre-qr2-8 --node-id pre-us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base --best-effort
gcloud alpha compute tpus queued-resources create pre-qr3-8 --node-id pre-us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base --best-effort
gcloud alpha compute tpus queued-resources create pre-qr4-8 --node-id pre-us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base --best-effort
gcloud alpha compute tpus queued-resources create pre-qr5-8 --node-id pre-us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base --best-effort
gcloud alpha compute tpus queued-resources create pre-qr6-8 --node-id pre-us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base --best-effort
gcloud alpha compute tpus queued-resources create pre-qr7-8 --node-id pre-us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base --best-effort
gcloud alpha compute tpus queued-resources create pre-qr8-8 --node-id pre-us-node-0 --project ai-memory --zone us-central2-b --accelerator-type v4-8 --runtime-version tpu-ubuntu2204-base --best-effort

gcloud alpha compute tpus queued-resources delete pre-qr1-8 --force --async


# Downloading and Processing climsim_high_res_train
# ssh into v4-8
# Build default env

pip install apache_beam 
pip install xarray huggingface_hub matplotlib netCDF4 h5py

git pull
rm -rf /home/joylunkad/.cache/huggingface/hub/datasets--LEAP--ClimSim_high-res/
for idx in $(seq 56 8 64); do
  python3 build_high_res_ds_local.py --write_initial_tfrecords --local_idx=$idx
done

screen -S high_res
git pull && python3 build_high_res_ds_local.py --finalize_shuffled_tfrecords --local_idx=1

git pull && python3 build_high_res_ds_local.py --shuffle_tfrecords --local_idx=-1

git pull && python3 build_high_res_ds_local.py --delete_tmp_files

git pull && python3 build_high_res_ds_local.py --shuffle_split_key=bucket_026
git pull && python3 build_high_res_ds_local.py --shuffle_split_key=bucket_072
git pull && python3 build_high_res_ds_local.py --shuffle_split_key=bucket_080
git pull && python3 build_high_res_ds_local.py --shuffle_split_key=bucket_088
git pull && python3 build_high_res_ds_local.py --shuffle_split_key=bucket_096 && python3 build_high_res_ds_local.py --shuffle_split_key=bucket_104

# Since 511 was already done, we only run the rest.
pip install apache_beam huggingface_hub NetCDF4 xarray h5py matplotlib 
git pull
for idx in $(seq 0 1 8); do
  tfds build --data_dir="gs://us2-climsim" --config_idx=$idx
done

for idx in $(seq 8 8 511); do
  tfds build --data_dir="gs://us2-climsim" --config_idx=$idx
  rm -rf /home/joylunkad/.cache/huggingface/hub/datasets--LEAP--ClimSim_high-res/
done


python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="source ~/.bashrc && echo $PATH"

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="source ~/.bashrc && git pull && python3 MaxText/train.py MaxText/configs/v4/25m_high_res.yml run_name=eval_last_1ep num_epochs=1 ici_data_parallelism=8"

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="source ~/.bashrc && git pull && python3 MaxText/train.py MaxText/configs/v4/25m_high_res.yml run_name=ep_10_lr=0.00005 num_epochs=10 ici_data_parallelism=8 n_checkpoints_every_epoch=2 n_evals_every_epoch=2 learning_rate=0.00005"


python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="source ~/.bashrc && git pull && python3 MaxText/train.py MaxText/configs/v4/25m_high_res.yml run_name=test_lr num_epochs=0.1 ici_data_parallelism=8"



gcloud alpha compute tpus queued-resources create pre-qr-1 --accelerator-type=v4-8 --runtime-version=tpu-ubuntu2204-base --node-count=2 --node-prefix=test  --best-effort
gcloud alpha compute tpus queued-resources list --filter=pre-qr-1
python3 multihost_runner.py --TPU_PREFIX=pre-us-node-2 --COMMAND="bash setup.sh"
python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="source ~/.bashrc && bash setup.sh MODE=stable JAX_VERSION=0.4.28"
python3 multihost_runner.py --TPU_PREFIX=pre-us-node-2 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml run_name=recreate_sub_test_32"
gcloud alpha compute tpus queued-resources delete pre-qr-1 --force --async

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml run_name=recreate_sub_test_32 dcn_data_parallelism=8"

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml run_name=test_cover_all_splits_ft num_epochs=0.5 per_device_batch_size=64 dtype='float32' checkpoint_period=800 learning_rate=5e-4 eval_interval=800 use_full_low_res_data=True cosine_learning_rate_final_fraction=0.001"

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml  run_name=test_arch_hr_8 num_epochs=0.1 n_evals_every_epoch=1000 n_checkpoints_every_epoch=500 use_high_res_data=True use_value_masks=True adam_b1=0.99 adam_b2=0.999 shuffle_buffer_multiplier=4 n_ds_per_split=128"

python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="python3 MaxText/train.py MaxText/configs/base.yml run_name=test_arch_lr_8 num_epochs=0.5 n_evals_every_epoch=100 n_checkpoints_every_epoch=50 use_full_low_res_data=True use_value_masks=True stack_inputs=True dropout_rate=0.1"


kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f test.csv
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f train.csv
unzip test.csv.zip


# Switch kaggle accounts ->
python3 multihost_runner.py --TPU_PREFIX=us-node-0 --COMMAND="rm -f ~/.kaggle/kaggle.json && gsutil cp gs://us2-climsim/lunkad_tv_kaggle.json ./ && cp lunkad_tv_kaggle.json ~/.kaggle/kaggle.json"

gsutil cp -AR gs://us2-climsim/maxtext/test_sw_base/checkpoints/48800 gs://us2-climsim/maxtext/test_sw_base/checkpoints/48800


# Clear files in /tmp with a certain prefix
rm -f /tmp/tmp_file_tensorflow_*