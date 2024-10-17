import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from scipy import io
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm
import torch.nn.functional as F
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_name', type=str)
parser.add_argument('--nSamples', type=int)
parser.add_argument('--latent_dim', type=int)
parser.add_argument('--file_flag', type=str)
opt = parser.parse_args()

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)


def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

def Com_ACC(data, clf5, is_svm=True):
    testData, testLabels = data
    
    testData = normalize(testData)
    if is_svm:
        pred = clf5.predict(testData)
    else:
        pred = torch.argmax(clf5(torch.from_numpy(testData).float().to(DEVICE)), dim=1)
        pred = pred.detach().cpu().numpy()

    correct_pred = np.sum((testLabels == pred))
    avgAcc = correct_pred/len(testLabels) 
    print("avgAcc:", avgAcc)

    f1 = f1_score(testLabels, pred, average='macro')   

    return avgAcc, f1


def Com_ACC_CA(data, clf5):
    testData, testLabels = data
    
    testData = normalize(testData)
    pred = clf5.predict(testData)
    # pred = torch.argmax(cl5(torch.from_numpy(testData).float().to(DEVICE)), dim=1)
    # pred = pred.detach().cpu().numpy()

    allTestClasses = sorted(list(set(testLabels.tolist())))
    dict_correct = {}
    dict_total = {}

    for ii in allTestClasses:
        dict_total[ii] = 0 
        dict_correct[ii] = 0

    for ii in range(0,testLabels.shape[0]):
        if(testLabels[ii] == pred[ii]):
            dict_correct[testLabels[ii]] = dict_correct[testLabels[ii]] + 1
        dict_total[testLabels[ii]] = dict_total[testLabels[ii]] + 1

        

    avgAcc = 0.0
    for ii in allTestClasses:
        avgAcc = avgAcc + (dict_correct[ii]*1.0)/(dict_total[ii])

    avgAcc = avgAcc/len(allTestClasses) 
    print('Average Class Accuracy = ' + str(avgAcc))

    return avgAcc

#===================================================================#
# Some Constants
m = 50 
n_x = 128
n_y = 128 
n_z = opt.latent_dim
interNo = n_x
nSamples = opt.nSamples

DEVICE = 'cuda'
DATA_DIR = f'./save_folder/{opt.data_name}_data_global{opt.file_flag}.zip'
# ============================================================ #
# VAE Model:

class UBCG_Model(nn.Module):
    def __init__(self, n_x, n_y, n_z, interNo) -> None:
        super(UBCG_Model, self).__init__()
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.n_z = int(n_z)
        self.interNo = int(interNo)

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(self.n_x + self.n_y, self.interNo), 
                                     nn.ReLU(), 
                                     nn.Dropout(p=0.3), 
                                     nn.Linear(self.interNo, self.interNo),
                                     nn.ReLU())
        
                
        self.mean = nn.Linear(self.interNo, n_z)
        self.log_sigma = nn.Linear(self.interNo, n_z)

        # Decoder
        self.decoder = nn.Sequential(nn.Linear(self.n_z + self.n_y, int(self.n_x/2)),
                                     nn.ReLU(),
                                     nn.Linear(int(self.n_x/2), self.n_x))
        
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    def sample_z(self, args):
        mu, log_sigma = args
        sigma = log_sigma.mul(0.5).exp_()
        eps = torch.FloatTensor(sigma.size()).normal_().to(DEVICE)
        return eps.mul(sigma).add_(mu)
    
    def generate_syn(self, cond):
        with torch.no_grad():
            z_sample_rand = torch.empty((cond.size(0), self.n_z)).normal_().to(DEVICE)
            z = torch.cat((z_sample_rand, cond), dim=1)
            x_gen = self.decoder(z)
        
        return x_gen
    
    def generate_syn_text(self, cond):
        with torch.no_grad():
            z_sample_rand = torch.empty((cond.size(0), self.n_z)).normal_().to(DEVICE)
            z = torch.cat((z_sample_rand, cond), dim=1)
            x_gen = self.decoder(z)

            text_z = torch.cat((z_sample_rand, x_gen), dim=1)
            text_gen = self.decoder(text_z)
        
        return x_gen, text_gen
    
    def forward(self, x, cond):
        x_cond = torch.cat((x, cond), dim=1)
        encode = self.encoder(x_cond)
        
        mean = self.mean(encode)
        log_var = self.log_sigma(encode)

        sampling = self.sample_z([mean, log_var])

        sampling_cond = torch.cat((sampling, cond), dim=1)
        x_recon = self.decoder(sampling_cond)

        return x_recon, sampling, mean, log_var 

# ============================================================ #
# Lodaing Data:

matcontent = torch.load(DATA_DIR)
attrs = normalize(matcontent['syn_class'])

test_feat = matcontent['final_node']
text_feat = matcontent['text_feat']
class_list = matcontent['task_lables_arr'].tolist()
test_label = np.array([class_list.index(i) for i in matcontent['final_gt']])
seen_classes = np.unique(test_label)

test_data = {'test_feat': test_feat,
             'test_text_feat': text_feat,
             'test_label': test_label,
             'test_label_global': matcontent['final_gt'],
             'class_list': class_list}
torch.save(test_data, './task_wise_syn_data/{}_test_data{}{}.zip'.format(opt.data_name, opt.task_id, opt.file_flag))

assert len(seen_classes) == 5
assert len(seen_classes) == len(attrs)

# ============================================================ #

model = UBCG_Model(n_x, n_y, n_z, interNo)
checkpoint = torch.load('./{}_global_model_neig{}.zip'.format(opt.data_name, opt.file_flag))
model.load_state_dict(checkpoint)
model.to(DEVICE)


model.eval()
for iter, cls in enumerate(seen_classes):
    idx_select = np.array([cls for _ in range(nSamples)])
    attr_select = attrs[idx_select] 
    # syn_sample = model.generate_syn(torch.from_numpy(attr_select).float().to(DEVICE))
    syn_sample_node, syn_sample_text = model.generate_syn_text(torch.from_numpy(attr_select).float().to(DEVICE))
    
    if iter == 0:
        syn_sample_node_global = syn_sample_node.detach().cpu().numpy()
        syn_sample_text_global = syn_sample_text.detach().cpu().numpy()
        syn_label = np.array([cls for _ in range(nSamples)])
    else:
        syn_sample_node_global = np.concatenate((syn_sample_node_global, syn_sample_node.detach().cpu().numpy()), axis=0)
        syn_sample_text_global = np.concatenate((syn_sample_text_global, syn_sample_text.detach().cpu().numpy()), axis=0)
        syn_label = np.concatenate((syn_label, np.array([cls for _ in range(nSamples)])), axis=0)


print('syn_sample_node_global:', syn_sample_node_global.shape)
print('syn_sample_text_global:', syn_sample_text_global.shape)
print('syn_label:', syn_label.shape)

task_data = {'syn_node_feat': syn_sample_node_global,
             'syn_text_feat': syn_sample_text_global,
             'syn_label': syn_label}

torch.save(task_data, './task_wise_syn_data/{}_task_data{}{}.zip'.format(opt.data_name, opt.task_id, opt.file_flag))















