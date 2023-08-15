#!/bin/bash
#SBATCH -o ./mocogrid.out # STDOUT
#SBATCH -e ./mocogrid.err # STDERR
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-node 1
#SBATCH --mem 256GB
#SBATCH -N 1
#SBATCH -p fvl
#SBATCH -t 2-00:00:00
#SBATCH -q medium
#SBATCH -J car-moco

tune_id=2
output_dir="/share/ckpt/cgn/vpt/output-mocov3-attention-grid"

datasets=("CUB" "OxfordFlowers" "StanfordCars" "StanfordDogs" "NABirds")
configs=("cub" "flowers" "cars" "dogs" "nabirds")
dataset=${datasets[${tune_id}]}
dataset_config=${configs[${tune_id}]}
model_root="/share/ckpt/cgn/vpt/model"
data_path="/share_io03_ssd/test2/cgn/${dataset}"

# Tune CUB with VPT:
for seed in "40"; do
    python tune_fgvc.py \
        --config-file configs/prompt/${dataset_config}.yaml \
        --train-type "prompt" \
        MODEL.TYPE "mocov3_vitb" \
        DATA.BATCH_SIZE 64 \
        MODEL.PROMPT.DEEP True \
        MODEL.PROMPT.DROPOUT 0.1 \
        MODEL.PROMPT.NUM_TOKENS 10 \
        MODEL.PROMPT.LOCATION "attention" \
        DATA.FEATURE "mocov3_vitb" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}" \
        OUTPUT_DIR "${output_dir}/seed${seed}"
done