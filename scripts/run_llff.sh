# export CUDA_VISIBLE_DEVICES=$1
dataset=/root/autodl-tmp/data/nerf_llff_data/fern
workspace=/root/autodl-tmp/data/output/llff/fern
lambda_dis_loss=$1


CUDA_VISIBLE_DEVICES=0 python train.py \
--source_path $dataset -m $workspace/dis=$lambda_dis_loss \
--eval  -r 8 --n_views 3 \
--random_background \
--iterations 10000 --position_lr_max_steps 10000 \
--densify_until_iter 10000 \
--densify_grad_threshold 0.0005 \
--gaussiansN 2 \
--coprune --coprune_threshold 5 \
--coreg --sample_pseudo_interval 1 --start_sample_pseudo 500\ 
--add_discriminator


CUDA_VISIBLE_DEVICES=0 python render.py \
--source_path $dataset -m $workspace/dis=$lambda_dis_loss \
--render_depth

CUDA_VISIBLE_DEVICES=0 python metrics.py \
--source_path $dataset -m $workspace/dis=$lambda_dis_loss \