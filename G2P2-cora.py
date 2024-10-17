import numpy as np
import sys

sys.path.append('../')
import os.path as osp
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
# import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model import CLIP, tokenize
from data_graph import DataHelper
from sklearn.metrics import accuracy_score, f1_score
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, preprocess_documents
from multitask import multitask_data_generator
from sklearn import preprocessing
import json

from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt



import sys
import csv
import os

def tsne_plot(data, label):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(data) 

    df = pd.DataFrame()
    df["y"] = label
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    plt.figure(figsize=(15,10))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(np.unique(label))),
                    data=df).set(title="CORA T-SNE projection")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    # plt.tight_layout()
    plt.savefig("seaborn_plot.png", format='png')

    
data_name = 'cora'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)
# device = torch.device("cpu")
num_nodes = 0
tit_list = []
lab_list = []
with open('data/train_text.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        # print('line is: ', line)
        # sys.exit()
        tit_list.append(line[2])
        lab_list.append(line[3])
        num_nodes += 1

# print('num_nodes', num_nodes)

labeled_ids = []
for i in range(len(lab_list)):
    # if lab_list[i] == 'nan':
    #     print(lab_list[i])
    if lab_list[i] != 'nan':
        labeled_ids.append(i)

# print('lab_id and lab_list: ', len(labeled_ids), len(lab_list))
# sys.exit()

print('{} nodes having lables'.format(len(labeled_ids)))

raw_edge_index = [[], []]
with open('data/mapped_edges.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        raw_edge_index[0].append(int(line[0]))
        raw_edge_index[1].append(int(line[1]))

edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
arr_edge_index = np.array(edge_index)
edge_index = np.array(edge_index)
edge_index = torch.from_numpy(edge_index).to(device)

# node_f = np.load('../cora/node_f_title.npy').astype(np.float32)
node_f = np.load('data/node_f.npy')
node_f = preprocessing.StandardScaler().fit_transform(node_f)
node_f = torch.from_numpy(node_f).to(device)

# print('node_f: ', node_f[labeled_ids].size())
# TSNE Plot
# tsne_plot(node_f[labeled_ids].cpu().numpy(), np.array(lab_list)[labeled_ids])
# sys.exit()


# label_texts = []
with open('data/lab_list.txt', 'r') as f:
    line = f.readline().strip().split('\t')
    label_texts = line

labels = []
for i in label_texts:
    if i != 'nan':
        labels.append(i)

# print('labels: ', len(labels))
# sys.exit()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

print("check different labels")
print('tit_list:', tit_list[0])
print('labeled_ids', labeled_ids[0])
print('lab_list', lab_list[0])
# sys.exit()

def main(args):
    setup_seed(seed)
    model = CLIP(args).to(device)

    model.load_state_dict(torch.load('./G2P2_datasets/pre-trained-model/{}/node_ttgt_8&12_0.1.pkl'.format(data_name), map_location=device))
    model.eval()
    
    
    checkpoint_org = torch.load('./org_seed_data/cora_org_data_seed{}'.format(seed))
    task_list, train_idx, val_idx, test_idx = checkpoint_org['task_list'], checkpoint_org['train_idx'], checkpoint_org['val_idx'], checkpoint_org['test_idx']
    
    
    task_data = {'task_list': task_list, 
                 'train_idx': train_idx, 
                 'val_idx': val_idx, 
                 'test_idx': test_idx}
        
    torch.save(task_data, './synthetic_data/cora/cora_task_data{}.zip'.format(args.file_flag))    

    acc_list = []
    f1_list = []
    for j in range(0, len(task_list)):        
        test_gt = np.array(lab_list)[np.array(test_idx[j])]
        test_idx_ts = torch.from_numpy(np.array(test_idx[j])).to(device)
        model.eval()
        task_lables_arr = np.array(labels)[task_list[j]]
        task_lables = task_lables_arr.tolist()
        
        task_prompt = []
        for a in range(len(task_lables)):
            prompt = the_template + task_lables[a]
            task_prompt.append(prompt)
        test_labels = tokenize(task_prompt, context_length=args.context_length).to(device)
        with torch.no_grad():
            syn_class = model.encode_text(test_labels)

        Data = DataHelper(arr_edge_index, args, test_idx[j])
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        node_feas = []

        with torch.no_grad():
            node_fea = model.encode_image(test_idx_ts, node_f, edge_index)
            node_feas.append(node_fea)

            text = np.array(tit_list)[test_idx_ts.cpu().numpy()].tolist()
            text_feat = tokenize(text, context_length=args.context_length).to(device)
            text_feat = model.encode_text(text_feat)

        node_feas = torch.cat(node_feas, dim=0)
        
        unnormailzed_node_feas = node_feas.detach().cpu().numpy() 
        unnormailzed_text_feas = text_feat.detach().cpu().numpy()
        unnormailzed_syn_class = syn_class.detach().cpu().numpy() 
        
        # Normalziing for cosine similarity
        syn_class /= syn_class.norm(dim=-1, keepdim=True)
        node_feas /= node_feas.norm(dim=-1, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        # Node and Text Embedding Fusion:
        similarity_node = (100.0 * node_feas @ syn_class.T).softmax(dim=-1)
        similarity_text = (100.0 * text_feat @ syn_class.T).softmax(dim=-1)
        similarity = args.lamda*similarity_node + (1-args.lamda)*similarity_text
        pred = similarity.argmax(dim=-1)
        score, _ = torch.max(similarity, dim=-1)
        
        pred = pred.cpu().numpy().reshape(-1)
        score = score.cpu().numpy().reshape(-1)
        y_pred = task_lables_arr[pred]

                
        save_dict = {'node_feat': unnormailzed_node_feas,
                     'text_feat': unnormailzed_text_feas,
                     'y_pred': y_pred,
                     'test_gt': test_gt,
                     'syn_class': unnormailzed_syn_class,
                     'task_lables_arr': task_lables_arr,
                     'final_node': unnormailzed_node_feas,
                     'final_gt': test_gt}
        
        torch.save(save_dict, './save_folder/cora_data_global{}.zip'.format(args.file_flag))
        
        del save_dict
        del node_feas
        del text_feat
        torch.cuda.empty_cache()

                
        acc = accuracy_score(test_gt, y_pred)
        acc_list.append(acc)
        f1 = f1_score(test_gt, y_pred, average='macro')
        f1_list.append(f1)

        print('acc is: ', acc)
        print('\n\n')
        
        # writing to csv file  
        if j == 0:
            with open('./results/cora_g2p2_result_neig_hybrid{}.csv'.format(args.file_flag), 'w') as csvfile:  
                # creating a csv writer object  
                csvwriter = csv.writer(csvfile)  
            
                # writing the fields  
                csvwriter.writerow([j, acc, f1]) 
        else:
            with open('./results/cora_g2p2_result_neig_hybrid{}.csv'.format(args.file_flag), 'a') as csvfile:  
                # creating a csv writer object  
                csvwriter = csv.writer(csvfile)  
            
                # writing the fields  
                csvwriter.writerow([j, acc, f1]) 

        os.system('sh UBCG_generate.sh {} {} {} {} {} {}'.format(j, seed, 'cora', args.nSamples, args.latent_dim, str(args.file_flag)))

                
    
    ans = round(np.mean(acc_list).item(), 4)
    print('zero shot acc', ans)
    sys.exit()
    all_acc_list[word].append(ans)

    ans = round(np.mean(f1_list).item(), 4)
    print('macro f1', ans)
    all_macf1_list[word].append(ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--hidden', type=str, default=16, help='number of hidden neurons')
    parser.add_argument('--epoch_num', type=int, default=101, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--neigh_num', type=int, default=3)
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    parser.add_argument('--edge_coef', type=float, default=0.1)
    
    parser.add_argument('--k_spt', type=int, default=5) # 0
    parser.add_argument('--k_val', type=int, default=5) # 0
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5) # 5

    parser.add_argument('--context_length', type=int, default=128)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)  # decided by the given vocab

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--nSamples', type=int, default=200)
    parser.add_argument('--latent_dim', type=int, default=50)
    parser.add_argument('--prompt_template', type=int, default=0)
    parser.add_argument('--file_flag', type=str, default='')

    # 2 heads, 6 layers, the first attempt
    # 8 heads, 12 layers, the second attempt

    args = parser.parse_args()

    start = time.perf_counter()
    all_acc_list = []
    all_macf1_list = []

    the_list_full = ['', 'a ', 'an ', 'a paper of ', 'a paper illustrating on the topic of ', 'a research paper on the topic of ']
    the_list = ['a ']
    for word in range(len(the_list)):
        all_acc_list.append([])
        all_macf1_list.append([])
        the_template = the_list[word]
        print('the_template=', the_template)
        print('\n')
        seed = int(math.pow(2, args.seed))
        print('seed', seed)
        main(args)
        print('\n')

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))

