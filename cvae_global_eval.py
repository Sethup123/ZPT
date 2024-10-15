import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from scipy import io
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn import svm
import torch.nn.functional as F
import csv
import argparse

np.random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=0)
parser.add_argument('--seed', type=int)
parser.add_argument('--data_name', type=str)
opt = parser.parse_args()


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

    return avgAcc


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

m = 50 #50
n_x = 128
n_y = 128 #312
n_z = 50
interNo = n_x
n_epoch = 25 #25
path = '../../Datasets/CUB/'
nSamples = 200
nTrain = 150 #150
nTest = 50 #50

DEVICE = 'cuda'
DATA_DIR = f'/mnt/data/sethupathy/G2P2-conditional-main/save_folder/{opt.data_name}_data_global.zip'
# ============================================================ #
# VAE Model:

class VAE_Model(nn.Module):
    def __init__(self, n_x, n_y, n_z, interNo) -> None:
        super(VAE_Model, self).__init__()
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

def VAE_Loss(x, x_recon, mean, log_var):
    l2_loss = torch.nn.MSELoss(reduction='none')
    recon_loss = l2_loss(x_recon, x)
    recon_loss = torch.mean(recon_loss, dim=1)
    kl = (0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
    
    return torch.mean(recon_loss), torch.mean(-kl), torch.mean(recon_loss - kl)

class Classifier(nn.Module):
    def __init__(self, in_dim=128, h_dim=512, out_dim=5) -> None:
        super(Classifier, self).__init__()
        # self.model = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, out_dim))
        self.model = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        out = self.model(x)
        return out

# ============================================================ #
# Lodaing Data:

matcontent = torch.load(DATA_DIR)
attrs = normalize(matcontent['syn_class'])

test_feat = matcontent['final_node']
class_list = matcontent['task_lables_arr'].tolist()
test_label = np.array([class_list.index(i) for i in matcontent['final_gt']])
seen_classes = np.unique(test_label)

test_data = {'test_feat': test_feat,
             'test_label': test_label,
             'test_label_global': matcontent['final_gt'],
             'class_list': class_list}
torch.save(test_data, './task_wise_syn_data/test_data{}.zip'.format(opt.task_id))

assert len(seen_classes) == 5
assert len(seen_classes) == len(attrs)

# ============================================================ #

model = VAE_Model(n_x, n_y, n_z, interNo)
checkpoint = torch.load('/mnt/data/sethupathy/CVAE/ZeroShot_CVAE-master/Disjoint/CUB/global_model.zip')
model.load_state_dict(checkpoint)
model.to(DEVICE)


model.eval()
syn_sample_global = []
for iter, cls in enumerate(seen_classes):
    idx_select = np.array([cls for _ in range(nSamples)])
    attr_select = attrs[idx_select] 
    syn_sample = model.generate_syn(torch.from_numpy(attr_select).float().to(DEVICE))
    
    if iter == 0:
        syn_sample_global = syn_sample.detach().cpu().numpy()
        syn_label = np.array([cls for _ in range(nSamples)])
    else:
        syn_sample_global = np.concatenate((syn_sample_global, syn_sample.detach().cpu().numpy()), axis=0)
        syn_label = np.concatenate((syn_label, np.array([cls for _ in range(nSamples)])), axis=0)


print('syn_sample_global:', syn_sample_global.shape)
print('syn_label:', syn_label.shape)

task_data = {'syn_node_feat': syn_sample_global,
             'syn_label': syn_label}

torch.save(task_data, './task_wise_syn_data/task_data{}.zip'.format(opt.task_id))

syn_sample_global = normalize(syn_sample_global, axis=1)

#################################################################################
# clf5_dataset = TensorDataset(torch.from_numpy(syn_sample_global).float(),
#                             torch.from_numpy(syn_label).long())
# clf5_dataloader = DataLoader(clf5_dataset, batch_size=512, shuffle=True)

# clf5 = Classifier()
# clf5.to(DEVICE)
# optim_cl5 = torch.optim.Adam(clf5.parameters(), lr=1e-3) #, weight_decay=0.001)



# clf5.train()
# for epoch in range(25):
#     print('train epoch: {}'.format(epoch))
#     for (x, y) in clf5_dataloader:
#         x, y = x.to(DEVICE), y.to(DEVICE)
#         pred = clf5(x)
#         loss = F.cross_entropy(pred, y)
#         optim_cl5.zero_grad()
#         loss.backward()
#         optim_cl5.step()
    
#     with torch.no_grad():
#         testData = np.array(test_feat)
#         testLabels = np.array(test_label)
#         seen_acc = Com_ACC([testData, testLabels], clf5, is_svm=False)

# clf5.eval()
#################################################################################

#################################################################################
print('Training SVM-100')
clf5 = svm.SVC(C=100, verbose=False)
clf5.fit(syn_sample_global, syn_label)
print('Predicting...')
#################################################################################


#################################################################################
# Seen
testData = np.array(test_feat)
testLabels = np.array(test_label)
seen_acc = Com_ACC([testData, testLabels], clf5, is_svm=True)

if opt.task_id == 0:
    with open('./{}_cvae_result_{}.csv'.format(opt.data_name, opt.seed), 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        csvwriter.writerow([seen_acc])
else:
    with open('./{}_cvae_result_{}.csv'.format(opt.data_name, opt.seed), 'a') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  

        # writing the fields  
        csvwriter.writerow([seen_acc])
#################################################################################















