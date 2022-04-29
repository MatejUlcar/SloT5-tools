#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=t5_sl_small
#SBATCH --gpus-per-node=4

export PYTHONPATH="/root:/text-to-text-transfer-transformer:/text-to-text-transfer-transformer/t5"
MODEL_DIR="/root/t5_sl_small"

srun --container-image ~/t5-container.sqsh --container-mount-home \
 t5_mesh_transformer  \
  --model_dir="${MODEL_DIR}" \
  --gin_file="/root/configs/dataset.gin" \
  --gin_file="/root/configs/t5.1.1.small.gin" \
  --gin_param="utils.run.mesh_shape = 'model:2,batch:2'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0','gpu:1','gpu:2','gpu:3']" \
  --gin_param="MIXTURE_NAME = 'mixture_slovene_test'" \
  --gin_param="utils.run.save_checkpoints_steps = 40000" \
  --gin_param="utils.run.keep_checkpoint_max = 2" \
  --gin_param="utils.run.train_steps = 1000000" \
  --module_import="mytask3"

