seed=(1 2 3 4 5)

for j in ${seed[@]}
do
    CUDA_VISIBLE_DEVICES=3 python UBCG_train_cora.py --data_name cora --file_flag _latent --seed $j --latent_dim 8

    CUDA_VISIBLE_DEVICES=3 python G2P2-cora.py --file_flag _latent --lamda 0.5 --seed $(expr $j - 1)  --latent_dim 8 
    CUDA_VISIBLE_DEVICES=3 python ZPT-cora_path.py --file_flag _latent --lamda 0.5 --seed $j 

    CUDA_VISIBLE_DEVICES=3 python log.py --file_flag _latent --value 1 --seed $j --data_name cora --file_name test_cora
done
