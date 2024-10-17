import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--value', type=str, help='value of hyperparameter to pass')
parser.add_argument('--seed', type=int, help='seed')
parser.add_argument('--data_name', type=str, help='name of dataset')
parser.add_argument('--file_name', type=str, help='name of file')
parser.add_argument('--file_flag', type=str, help='name of file')
args = parser.parse_args()

# datetime object containing current date and time
now = datetime.now() 
# dd/mm/YY H:M:S
dt_string = now.strftime("%b-%d-%Y %H:%M:%S")


# G2P2 Result:
path = './results/{}_g2p2_result_neig_hybrid{}.csv'.format(args.data_name, args.file_flag) #  {}_g2p2_result_neig_hybrid.csv
result = pd.read_csv(path, sep=',', header=None)
g2p2_result = np.mean(result.values[:,1:], axis=0)
# print(g2p2_result)

# G2P2_coop Result:
path = './results/{}_coop_result_neig_hybrid{}.csv'.format(args.data_name, args.file_flag)
result = pd.read_csv(path, sep=',', header=None)
g2p2_coop_result = np.mean(result.values[:,1:], axis=0)
# print(g2p2_coop_result)

with open('./results_final/{}_{}_{}_{}.csv'.format(args.data_name, args.file_name, args.file_flag, args.seed), mode='a') as file:  # num_neighbour_no_hybrid_no_desc_seed
    writer = csv.writer(file)
    writer.writerow([dt_string, 'acc', args.value, g2p2_result[0], g2p2_coop_result[0]])
    writer.writerow([dt_string, 'f1', args.value, g2p2_result[1], g2p2_coop_result[1]])



