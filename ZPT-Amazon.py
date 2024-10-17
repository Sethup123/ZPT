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
from multitask_amazon import multitask_data_generator
from model_g_coop_hybrid import CoOp_synthetic
import json
from data_graph import DataHelper
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize

import sys
import csv



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
    clip_model.load_state_dict(torch.load('./G2P2_datasets/pre-trained-model/{}/node_ttgt_8&12_10.pkl'.format(args.data_name), map_location=device))
    
  
    checkpoint = torch.load('./synthetic_data/cora/{}_task_data{}.zip'.format(args.data_name, args.file_flag))
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

        checkpoint_syn = torch.load('./task_wise_syn_data/{}_task_data{}{}.zip'.format(args.data_name, j, args.file_flag))
        node_feat = torch.from_numpy(checkpoint_syn['syn_node_feat']).cuda()
        text_feat = torch.from_numpy(checkpoint_syn['syn_text_feat']).cuda()
        node_label = torch.from_numpy(checkpoint_syn['syn_label']).cuda()

        print('checkpoint node_feat', node_feat.size())
        print('checkpoint text_feat', text_feat.size())
        print('checkpoint node_label', node_label.size())

        best_val = 0
        patience = 10
        counter = 0

        train_dataset = TensorDataset(node_feat, text_feat, node_label)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        for epoch in range(1, args.ft_epoch + 1):
            model.train()
            for itr, (node_feat, text_feat, node_label) in enumerate(train_dataloader):
                train_logits = model.forward(node_feat, text_feat, node_label)

            model.eval()
            with torch.no_grad():
                res = model.forward(node_feat, text_feat, node_label, training=False)
                val_acc = accuracy_score(node_label.cpu(), res.argmax(dim=1).cpu())                
                if val_acc <= best_val:
                    counter += 1
                    if counter >= patience:
                        break
                else:
                    best_val = val_acc
                    torch.save(model, './res/{}/g_coop{}.pkl'.format(args.data_name, args.file_flag))
                    counter = 0
        
        best_model = torch.load('./res/{}/g_coop{}.pkl'.format(args.data_name, args.file_flag))
        best_model.eval()
        
        
        checkpoint_test_data = torch.load('./task_wise_syn_data/{}_test_data{}{}.zip'.format(args.data_name, j, args.file_flag))
        test_node_feat = torch.from_numpy(normalize(checkpoint_test_data['test_feat'])).cuda()
        test_text_feat = torch.from_numpy(normalize(checkpoint_test_data['test_text_feat'])).cuda()
        test_node_label = torch.from_numpy(checkpoint_test_data['test_label']).cuda()
        

        with torch.no_grad():
            res = model.forward(test_node_feat, test_text_feat, test_node_label, training=False)
            test_acc = accuracy_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu())
            all_acc.append(test_acc)
            f1 = f1_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu(), average='macro')
            f1_list.append(f1)

        if j == 0:
            with open('./results/{}_coop_result_neig_hybrid{}.csv'.format(args.data_name, args.file_flag), 'w') as csvfile:  
                # creating a csv writer object  
                csvwriter = csv.writer(csvfile)  
            
                # writing the fields  
                csvwriter.writerow([j, test_acc, f1])
        else:
            with open('./results/{}_coop_result_neig_hybrid{}.csv'.format(args.data_name, args.file_flag), 'a') as csvfile:  
                # creating a csv writer object  
                csvwriter = csv.writer(csvfile)  
            
                # writing the fields  
                csvwriter.writerow([j, test_acc, f1])

        # sys.exit()
    
    ans = round(np.mean(all_acc).item(), 4)
    print('acc', ans)

    ans = round(np.mean(f1_list).item(), 4)
    print('macro f1', ans)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--ft_epoch', type=int, default=1, help='fine-tune epoch') # 50
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--batch_size', type=int, default=1000) #1000
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
    parser.add_argument('--prompt_lr', type=float, default=0.01)

    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=False)
    parser.add_argument('--ctx_init', type=bool, default=False) #True

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--data_name', type=str, default="Industrial_and_Scientific") #"Musical_Instruments"  "Arts_Crafts_and_Sewing"
    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--seed', type=int)

    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--file_flag', type=str, default='')


    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)

    num_nodes = 0
    tit_list = []
    tit_dict = json.load(open('./G2P2_datasets/{}_text.json'.format(args.data_name)))
    new_dict = {}

    for i in range(len(tit_dict)):
        num_nodes += 1
        new_dict[i] = tit_dict[str(i)]

    print('num_nodes', num_nodes)

    edge_index = np.load('./G2P2_datasets/{}_edge.npy'.format(args.data_name))

    arr_edge_index = edge_index

    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load('./G2P2_datasets/{}_f_m.npy'.format(args.data_name))
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)

    id_lab_dict = json.load(open('./G2P2_datasets/{}_id_labels.json'.format(args.data_name)))
    id_lab_list = sorted(id_lab_dict.items(), key=lambda d: int(d[0]))

    labeled_ids = []
    lab_list = []
    for i in id_lab_list:
        if i[1] != 'nan' or i[1] != '' or i[1] != ' ':
            labeled_ids.append(int(i[0]))
            lab_list.append(i[1])

    labels = sorted(list(set(lab_list)))

    start = time.perf_counter()
    all_acc_list = []
    all_macf1_list = []

    seed = args.seed  # 1
    print('seed', seed)
    main(args)
    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
