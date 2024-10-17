set -e


# Dataset Names: Musical Instruments, Industrial_and_Scientific, Arts_Crafts_and_Sewing

seed=(1 2 3 4 5)
CUDA_VISIBLE_DEVICES=2 python UBCG_train_Amazon.py --file_flag _latent --data_name Musical_Instruments --seed 1 --latent_dim 8
    
for j in ${seed[@]}
do 
    CUDA_VISIBLE_DEVICES=2 python G2P2-Amazon.py --file_flag _latent --data_name Musical_Instruments --lamda 0.5 --seed $(expr $j - 1)  --latent_dim 8 
    CUDA_VISIBLE_DEVICES=2 python ZPT-Amazon.py --file_flag _latent --data_name Musical_Instruments --lamda 0.5 --seed $j

    CUDA_VISIBLE_DEVICES=2 python log.py --file_flag _latent --value 1 --seed $j --data_name Musical_Instruments --file_name MI_final 
done




