#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out

module load python/3.10
module load cuda/12.2
module load xtb
cd ..
source .venv/bin/activate
cd src

TORCH_NCCL_BLOCKING_WAIT=0 CUDA_VISIBLE_DEVICES=0 python train_condgenerator.py --dataset_name jmt_cont_core --num_layers 3 --tag jmt_cont_core --bf16 --check_sample_every_n_epoch 999 --save_every_n_epoch 1 --dropout 0.0 --warmup_steps 100 --lr_decay 0.1 --beta2 0.95 --weight_decay 0.1 --lambda_predict_prop 1.0 --lambda_predict_prop_always --batch_size 64 --lr 2.5e-4 --max_epochs 1000 --n_gpu 1 --randomize_order --start_random --scaling_type minmax --special_init --nhead 16 --swiglu --expand_scale 2.0 --max_len 700 --gpt --no_bias --rmsnorm --rotary --limited_properties --not_allow_empty_bond --test --ood_values 1 1 2.6 2.8 0 0.2 --ood_names target_core aS1 adelta --best_out_of_k 1 --guidance_min 1.5 1.5 1.5 --guidance_max 1.5 1.5 1.5 --only_ood "$@"

