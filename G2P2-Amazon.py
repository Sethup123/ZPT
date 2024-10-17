import sys

sys.path.append('../')
import os.path as osp
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import numpy as np
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
from multitask_amazon import multitask_data_generator
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

from tqdm import trange, tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
parser.add_argument('--hidden', type=str, default=16, help='number of hidden neurons')
parser.add_argument('--epoch_num', type=int, default=101, help='epoch number')
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--neigh_num', type=int, default=3)

parser.add_argument('--gnn_input', type=int, default=128)
# parser.add_argument('--gnn_input', type=int, default=300)
parser.add_argument('--gnn_hid', type=int, default=128)
parser.add_argument('--gnn_output', type=int, default=128)
parser.add_argument('--edge_coef', type=float, default=0.1)
# parser.add_argument('--lamda', type=float, default=0.5)

parser.add_argument('--k_spt', type=int, default=5)
parser.add_argument('--k_val', type=int, default=5)
parser.add_argument('--k_qry', type=int, default=50)
parser.add_argument('--n_way', type=int, default=5)

parser.add_argument('--context_length', type=int, default=128) #120

parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--transformer_heads', type=int, default=8)
parser.add_argument('--transformer_layers', type=int, default=12)
parser.add_argument('--transformer_width', type=int, default=512)
parser.add_argument('--vocab_size', type=int, default=49408)  # decided by the given vocab

parser.add_argument('--data_name', type=str, default="Industrial_and_Scientific") #"Musical_Instruments"  "Arts_Crafts_and_Sewing"
parser.add_argument('--file_flag', type=str, default='')

# 2 heads, 6 layers, the first attempt
# 8 heads, 12 layers, the second attempt

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--lamda', type=float, default=0.5)
parser.add_argument('--nSamples', type=int, default=200)
parser.add_argument('--latent_dim', type=int, default=50)
args = parser.parse_args()




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


data_name = args.data_name #'Musical_Instruments' #'Arts_Crafts_and_Sewing' #'Musical_Instruments' #'Industrial_and_Scientific'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)
# device = torch.device("cpu")
FType = torch.FloatTensor
LType = torch.LongTensor


id_lab_dict = json.load(open('./G2P2_datasets/{}_id_labels.json'.format(data_name)))
id_lab_list = sorted(id_lab_dict.items(), key=lambda d: int(d[0]))

labeled_ids = []
lab_list = []

for i in id_lab_list:
    if i[1] != 'nan' or i[1] != '' or i[1] != ' ':
        labeled_ids.append(int(i[0]))
        lab_list.append(i[1])

print('number of labels', len(lab_list))

edge_index = np.load('./G2P2_datasets/{}_edge.npy'.format(data_name))

arr_edge_index = edge_index

edge_index = torch.from_numpy(edge_index).to(device)

node_f = np.load('./G2P2_datasets/{}_f_m.npy'.format(data_name))
node_f = preprocessing.StandardScaler().fit_transform(node_f)
node_f = torch.from_numpy(node_f)

labels = sorted(list(set(lab_list)))

print('labels[:10]', labels[:10])
print('labels[-10:]', labels[-10:])
print('number of lables', len(labels))


num_nodes = 0
tit_list = []
tit_dict = json.load(open('./G2P2_datasets/{}_text.json'.format(data_name)))
new_dict = {}

for i in range(len(tit_dict)):
    num_nodes += 1
    new_dict[i] = tit_dict[str(i)]
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True



def main(args):
    setup_seed(seed)
    model = CLIP(args).to(device)

    model.load_state_dict(torch.load('./G2P2_datasets/pre-trained-model/{}/node_ttgt_8&12_10.pkl'.format(data_name), map_location=device))
    model.eval()

    global node_f
    node_f = node_f.to(device)
    node_embed = model.encode_image(np.arange(node_f.size(0)), node_f, edge_index)
    node_embed = node_embed.detach().cpu().numpy()
    

    
    checkpoint_org = torch.load('./org_seed_data/{}_org_data_seed{}'.format(data_name, seed))
    task_list, train_idx, val_idx, test_idx = checkpoint_org['task_list'], checkpoint_org['train_idx'], checkpoint_org['val_idx'], checkpoint_org['test_idx']
    
    
    task_data = {'task_list': task_list, 
                 'train_idx': train_idx, 
                 'val_idx': val_idx, 
                 'test_idx': test_idx}
        
    torch.save(task_data, './synthetic_data/cora/{}_task_data{}.zip'.format(data_name, args.file_flag))

    acc_list = []
    f1_list = []
    for j in trange(len(task_list)):  

        test_idx_ts = torch.from_numpy(np.array(test_idx[j])).to(device)
        test_gt = []
        for a in test_idx[j]:
            test_gt.append(id_lab_dict[str(a)])
        
        task_lables_arr = np.array(labels)[task_list[j]]
        task_lables = task_lables_arr.tolist()

        task_prompt = []
        for a in range(len(task_lables)):
            prompt = the_template + ' ' + task_lables[a]
            task_prompt.append(prompt)
        
        test_labels = tokenize(task_prompt, context_length=args.context_length).to(device)
        with torch.no_grad():
            syn_class = model.encode_text(test_labels)

        Data = DataHelper(arr_edge_index, args, test_idx[j])
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        node_feas = []
        text_feat = []
        for i_batch, sample_batched in enumerate(loader):
            s_n = sample_batched['s_n'].numpy()
            t_n = sample_batched['t_n'].numpy()
            with torch.no_grad():
                node_fea = model.encode_image(s_n, node_f, edge_index)
                node_feas.append(node_fea)

                text = [new_dict[i] for i in s_n]
                text_fea = tokenize(text, context_length=args.context_length).to(device)
                text_feat.append(model.encode_text(text_fea))
        node_feas = torch.cat(node_feas, dim=0)
        text_feat = torch.cat(text_feat, dim=0)
        
                
        unnormailzed_node_feas = node_feas.detach().cpu().numpy() 
        unnormailzed_text_feas = text_feat.detach().cpu().numpy()
        unnormailzed_syn_class = syn_class.detach().cpu().numpy() 
        
        # Saving node features for visualization
        torch.save({'node_feat': node_feas,
                    'text_feat': text_feat,
                    'node_label': test_gt}, './data_viz/{}_test_node_feat_global{}.zip'.format(data_name, args.file_flag))


        # Normalizing for cosine similarity
        syn_class /= syn_class.norm(dim=-1, keepdim=True)
        node_feas /= node_feas.norm(dim=-1, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        # Node Text Hybrid Fusion
        similarity_node = (100.0 * node_feas @ syn_class.T).softmax(dim=-1)
        similarity_text = (100.0 * text_feat @ syn_class.T).softmax(dim=-1)
        similarity = args.lamda*similarity_node + (1-args.lamda)*similarity_text
        pred = similarity.argmax(dim=-1)
        score, _ = torch.max(similarity, dim=-1)
        score = score.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy().reshape(-1)
        y_pred = task_lables_arr[pred]
        
                
        save_dict = {'node_feat': unnormailzed_node_feas,
                     'text_feat': unnormailzed_text_feas,
                     'y_pred': y_pred,
                     'test_gt': test_gt,
                     'syn_class': unnormailzed_syn_class,
                     'task_lables_arr': task_lables_arr,
                     'final_node': unnormailzed_node_feas,
                     'final_gt': test_gt}
        
        torch.save(save_dict, './save_folder/{}_data_global{}.zip'.format(data_name, args.file_flag))
        
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
            with open('./results/{}_g2p2_result_neig_hybrid{}.csv'.format(data_name, args.file_flag), 'w') as csvfile:  
                # creating a csv writer object  
                csvwriter = csv.writer(csvfile)  
            
                # writing the fields  
                csvwriter.writerow([j, acc, f1]) 
        else:
            with open('./results/{}_g2p2_result_neig_hybrid{}.csv'.format(data_name, args.file_flag), 'a') as csvfile:  
                # creating a csv writer object  
                csvwriter = csv.writer(csvfile)  
            
                # writing the fields  
                csvwriter.writerow([j, acc, f1]) 

        os.system('sh UBCG_generate.sh {} {} {} {} {} {}'.format(j, seed, data_name, args.nSamples, args.latent_dim, args.file_flag))

    ans = round(np.mean(acc_list).item(), 4)
    print('zero shot acc', ans)
    all_acc_list[word].append(ans)

    ans = round(np.mean(f1_list).item(), 4)
    print('macro f1', ans)
    all_macf1_list[word].append(ans)


if __name__ == '__main__':
    
    start = time.perf_counter()
    all_acc_list = []
    all_macf1_list = []

    if args.data_name == 'Musical_Instruments':
        the_list = ['musical ']
    if args.data_name == 'Industrial_and_Scientific':
        the_list = ['an industrial and scientific ']
    if args.data_name == 'Arts_Crafts_and_Sewing':
        the_list = ['a ']
    
    for word in range(len(the_list)):
        all_acc_list.append([])
        all_macf1_list.append([])
        the_template = the_list[word]
        print('the_template=', the_template)
        print('\n')
        
        # for jack in range(0,1):
        seed = int(math.pow(2, args.seed))
        print('seed', seed)
        main(args)
        print('\n')

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))

