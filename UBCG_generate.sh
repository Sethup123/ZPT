set -e
CUDA_VISIBLE_DEVICES=3 python UBCG_generate.py --task_id $1 --seed $2 --data_name $3 --nSamples $4 --latent_dim $5 --file_flag $6