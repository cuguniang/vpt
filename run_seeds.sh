#!/bin/bash
#SBATCH -o ./out_seeds.out # STDOUT
#SBATCH -e ./err_seeds.err # STDERR
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node 1
#SBATCH --mem 256GB
#SBATCH -N 1
#SBATCH -p fvl
#SBATCH -t 2-00:00:00
#SBATCH -J h-fl
#SBATCH -q medium

tune_id=1
output_dir="/share/ckpt/cgn/vpt/output-mae-attention-seeds"

datasets=("CUB" "OxfordFlowers" "StanfordCars" "StanfordDogs" "NABirds")
configs=("cub" "flowers" "cars" "dogs" "nabirds")
dataset=${datasets[${tune_id}]}
dataset_config=${configs[${tune_id}]}
model_root="/share/ckpt/cgn/vpt/model"
data_path="/share_io03_ssd/test2/cgn/${dataset}"

# lr = base_lr / 256 * cfg.DATA.BATCH_SIZE
for seed in "40" "42" "314" "511" "666" "800" "2023" "13" "25" "4647" "197" "768" "314100"; do
     python train.py \
        --config-file configs/prompt/${dataset_config}.yaml \
        MODEL.TYPE "ssl-vit" \
        DATA.BATCH_SIZE 64 \
        SOLVER.BASE_LR 0.125 \
        SOLVER.WEIGHT_DECAY 0.0001 \
        MODEL.PROMPT.DEEP True \
        MODEL.PROMPT.DROPOUT 0.1 \
        MODEL.PROMPT.LOCATION "attention" \
        MODEL.PROMPT.NUM_TOKENS 10 \
        DATA.FEATURE "mae_vitb16" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}" \
        OUTPUT_DIR "${output_dir}/seed${seed}"
     
done