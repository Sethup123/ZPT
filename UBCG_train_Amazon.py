import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from scipy import io
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn import svm
import torch.nn.functional as F
import argparse
import json


parser = argparse.ArgumentParser()
# parser.add_argument('--task_id', type=int, default=0)
# parser.add_argument('--seed', type=int)
parser.add_argument('--data_name', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=50)
parser.add_argument('--file_flag', type=str, default='')
opt = parser.parse_args()

####################################################
# orginal seed setup
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
####################################################



def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

def Com_ACC(data, clf5):
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
n_epoch = 25 

DEVICE = 'cuda'
DATA_DIR = f'./save_folder/{opt.data_name}_node_text_emb_neig.zip'
# ============================================================ #
# UBGC Model:

class UBGC_Model(nn.Module):
    def __init__(self, n_x, n_y, n_z, interNo) -> None:
        super(UBGC_Model, self).__init__()
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
    
    def forward(self, x, cond):
        x_cond = torch.cat((x, cond), dim=1)
        encode = self.encoder(x_cond)
        
        mean = self.mean(encode)
        log_var = self.log_sigma(encode)

        sampling = self.sample_z([mean, log_var])

        sampling_cond = torch.cat((sampling, cond), dim=1)
        x_recon = self.decoder(sampling_cond)

        return x_recon, sampling, mean, log_var 

def UBGC_Loss(x, x_recon, mean, log_var):
    l2_loss = torch.nn.MSELoss(reduction='none')
    recon_loss = l2_loss(x_recon, x)
    recon_loss = torch.mean(recon_loss, dim=1)
    kl = (0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
    
    return torch.mean(recon_loss), torch.mean(-kl), torch.mean(recon_loss - kl)

# ============================================================ #
# Lodaing Data:

id_lab_dict = json.load(open('./G2P2_datasets/{}_id_labels.json'.format(opt.data_name)))
id_lab_list = sorted(id_lab_dict.items(), key=lambda d: int(d[0]))

labeled_ids = []
lab_list = []

for i in id_lab_list:
    if i[1] != 'nan' or i[1] != '' or i[1] != ' ':
        labeled_ids.append(int(i[0]))
        lab_list.append(i[1])

labeled_ids = np.array(labeled_ids)
print('labeled IDs are...', labeled_ids.shape)


matcontent = torch.load(DATA_DIR)

core_text = matcontent['text_emb']
attrs = matcontent['text_emb']

print('attrs shape:', attrs.shape)

train_feat = normalize(matcontent['node_emb'][labeled_ids])
train_cond = normalize(np.array(attrs)[labeled_ids])
core_text = normalize(core_text[labeled_ids])


assert len(train_feat) == len(train_cond)

train_dataset = TensorDataset(torch.from_numpy(train_feat).float(), 
                              torch.from_numpy(train_cond).float(),
                              torch.from_numpy(core_text).float())

train_dataloader = DataLoader(train_dataset, batch_size=m, shuffle=True)
# ============================================================ #

model = UBGC_Model(n_x, n_y, n_z, interNo)
model.to(DEVICE)

optim = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(n_epoch):
    loss_list = {'recon': [],
                 'kl': [],
                 'loss': []}
    
    for (x, cond, core_text) in train_dataloader:
        x, cond, core_text = x.to(DEVICE), cond.to(DEVICE), core_text.to(DEVICE)
        x_recon_node, sampling_node, mean_node, log_var_node = model(x, cond)
        recon_node, kl_node, loss_node = UBGC_Loss(x, x_recon_node, mean_node, log_var_node)

        x_recon_text, sampling_text, mean_text, log_var_text = model(cond, x)
        recon_text, kl_text, loss_text = UBGC_Loss(cond, x_recon_text, mean_text, log_var_text)
        
        loss = loss_node + loss_text
        
        loss_list['recon'].append([recon_node.item(), recon_text.item()])
        loss_list['kl'].append([kl_node.item(), kl_text.item()])
        loss_list['loss'].append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        
            
    avg_recon = np.mean(np.array(loss_list["recon"]), axis=0)
    avg_kl = np.mean(np.array(loss_list["kl"]), axis=0)
    avg_loss = np.mean(np.array(loss_list["loss"]))
    
    print(f'Epoch: {epoch}  recon: {avg_recon}   kl: {avg_kl}  loss: {avg_loss}')
    


model.eval()
torch.save(model.state_dict(), './{}_global_model_neig{}.zip'.format(opt.data_name, opt.file_flag))










