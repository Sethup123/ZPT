import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model import CLIP, tokenize
from torch import nn, optim
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
# from multitask_2 import multitask_data_generator
from multitask import multitask_data_generator
from model_g_coop_hybrid import CoOp_synthetic
import json
from data_graph import DataHelper
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize

import sys
import csv
from torch.utils.data import TensorDataset, DataLoader




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(seed)

    clip_model = CLIP(args)
    clip_model.load_state_dict(torch.load('./G2P2_datasets/pre-trained-model/{}/node_ttgt_8&12_0.1.pkl'.format(data_name), map_location=device))

   
    checkpoint = torch.load('./synthetic_data/cora/cora_task_data{}.zip'.format(args.file_flag))
    task_list, train_idx, val_idx, test_idx = checkpoint['task_list'], checkpoint['train_idx'], checkpoint['val_idx'], checkpoint['test_idx']
    
    
    all_acc = []
    f1_list = []
    for j in range(len(task_list)):

        task_lables_arr = np.array(labels)[task_list[j]]
        task_lables = task_lables_arr.tolist()
        
        Data = DataHelper(arr_edge_index, args, val_idx[j])
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for i_batch, sample_batched in enumerate(loader):
            s_n = sample_batched['s_n'].numpy()
            t_n = sample_batched['t_n'].numpy()

        model = CoOp_synthetic(args, task_lables, clip_model, None, device)

        checkpoint_syn = torch.load('./task_wise_syn_data/cora_task_data{}{}.zip'.format(j, args.file_flag))
        node_feat = torch.from_numpy(checkpoint_syn['syn_node_feat']).cuda()
        text_feat = torch.from_numpy(checkpoint_syn['syn_text_feat']).cuda()
        node_label = torch.from_numpy(checkpoint_syn['syn_label']).cuda()

        best_val = 0
        patience = 10
        counter = 0

        train_dataset = TensorDataset(node_feat, text_feat, node_label)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # Zero-Shot Prompt Tuning
        for epoch in range(1, args.ft_epoch + 1):
            model.train()
            for itr, (node_feat, text_feat, node_label) in enumerate(train_dataloader):
                train_logits = model.forward(node_feat, text_feat, node_label)

            model.eval()
            with torch.no_grad():
                res = model.forward(node_feat, text_feat, node_label, training=False)
                val_acc = accuracy_score(node_label.cpu(), res.argmax(dim=1).cpu())
                print('epoch: {}  val_acc: {}'.format(epoch, val_acc))
                if val_acc <= best_val:
                    counter += 1
                    if counter >= patience:
                        break
                else:
                    best_val = val_acc
                    torch.save(model, './res/{}/g_coop{}.pkl'.format(data_name, args.file_flag))
                    counter = 0


                best_model = torch.load('./res/{}/g_coop{}.pkl'.format(data_name, args.file_flag))
                best_model.eval()

                
                checkpoint_test_data = torch.load('./task_wise_syn_data/cora_test_data{}{}.zip'.format(j, args.file_flag))
                test_node_feat = torch.from_numpy(normalize(checkpoint_test_data['test_feat'])).cuda()
                test_text_feat = torch.from_numpy(normalize(checkpoint_test_data['test_text_feat'])).cuda()
                test_node_label = torch.from_numpy(checkpoint_test_data['test_label']).cuda()
                

                res = model.forward(test_node_feat, test_text_feat, test_node_label, training=False)
                test_acc = accuracy_score(test_node_label.cpu(), res.argmax(dim=1).cpu())
                print('test acc prompt tunin: ', test_acc)

        best_model = torch.load('./res/{}/g_coop{}.pkl'.format(data_name, args.file_flag))
        best_model.eval()

        

        checkpoint_test_data = torch.load('./task_wise_syn_data/cora_test_data{}{}.zip'.format(j, args.file_flag))
        test_node_feat = torch.from_numpy(normalize(checkpoint_test_data['test_feat'])).cuda()
        test_text_feat = torch.from_numpy(normalize(checkpoint_test_data['test_text_feat'])).cuda()
        test_node_label = torch.from_numpy(checkpoint_test_data['test_label']).cuda()
        

        with torch.no_grad():
            res = model.forward(test_node_feat, test_text_feat, test_node_label, training=False)
            test_acc = accuracy_score(test_node_label.cpu(), res.argmax(dim=1).cpu())
            print('test acc prompt tunin: ', test_acc)
            all_acc.append(test_acc)
            f1 = f1_score(test_node_label.cpu(), res.argmax(dim=1).cpu(), average='macro')
            f1_list.append(f1)

        if j == 0:
            with open('./results/cora_coop_result_neig_hybrid{}.csv'.format(args.file_flag), 'w') as csvfile:  
                # creating a csv writer object  
                csvwriter = csv.writer(csvfile)  
            
                # writing the fields  
                csvwriter.writerow([j, test_acc, f1])
        else:
            with open('./results/cora_coop_result_neig_hybrid{}.csv'.format(args.file_flag), 'a') as csvfile:  
                # creating a csv writer object  
                csvwriter = csv.writer(csvfile)  
            
                # writing the fields  
                csvwriter.writerow([j, test_acc, f1])


    ans = round(np.mean(all_acc).item(), 4)
    print('acc', ans)

    ans = round(np.mean(f1_list).item(), 4)
    print('macro f1', ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--ft_epoch', type=int, default=1, help='fine-tune epoch') # 50
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--batch_size', type=int, default=64) # 64
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)

    parser.add_argument('--edge_coef', type=float, default=0.1)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)

    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--coop_n_ctx', type=int, default=4)
    parser.add_argument('--prompt_lr', type=float, default=0.01) #0.01

    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=False)
    parser.add_argument('--ctx_init', type=bool, default=False)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--file_flag', type=str, default='')

    args = parser.parse_args()

    data_name = 'cora'
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)
    # device = torch.device("cpu")
    FType = torch.FloatTensor
    LType = torch.LongTensor

    num_nodes = 0
    tit_list = []
    lab_list = []
    with open('./data/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[2])
            lab_list.append(line[3])
            num_nodes += 1

    print('num_nodes', num_nodes)

    labeled_ids = []
    for i in range(len(lab_list)):
        if lab_list[i] != 'nan':
            labeled_ids.append(i)

    print('{} nodes having lables'.format(len(labeled_ids)))

    raw_edge_index = [[], []]
    with open('./data/mapped_edges.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            raw_edge_index[0].append(int(line[0]))
            raw_edge_index[1].append(int(line[1]))

    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    arr_edge_index = np.array(edge_index)
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load('./data/node_f.npy')
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)

    # label_texts = []
    with open('./data/lab_list.txt', 'r') as f:
        line = f.readline().strip().split('\t')
        label_texts = line

    labels = []
    for i in label_texts:
        if i != 'nan':
            labels.append(i)

    start = time.perf_counter()
    all_acc_list = []
    all_macf1_list = []

    seed = args.seed
    print('seed', seed)
    main(args)
    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
