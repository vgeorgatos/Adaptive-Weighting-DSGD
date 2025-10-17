import torch
from torchvision import datasets, transforms
import numpy as np
#from torchsummary import summary
#from tqdm import tqdm
import matplotlib.pyplot as plt
import copy


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset.partition import FMNISTPartitioner
from fedlab.utils.functional import partition_report

#import partition_data as pd
import os

hist_color = '#4169E1'


'Helper functions that create non-IID data partitions used in Federated/Distributed Learning. Especially, the codes use the FedLab data partitoners'
'for creating non-IID partitions (e.g. Shard, Dirichlet, IID,...) for common datasets like CIFAR-10/100, (Fashion)MNIST, etc.' 
'For more info about FedLab see https://github.com/SMILELab-FL/FedLab.'


def CIFAR10Partition(num_clients, da):

    'da: dirichlet_alpha'
    'num_clients: number of clients'
    'seed: random seed'
    'This function creates the Non-IID data partitions using the Dirichlet distribution method from the fedlab library'

    'Returns: a list containing the dataloaders for each client and the testloader'
    
    train_tran =   transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_tran  = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    num_classes = 10
    #seed = 2019
    trainset = datasets.CIFAR10(root="./data",
                                            train=True, download=True,transform= train_tran)




    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_tran)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                            shuffle=False, num_workers=2, persistent_workers=True, prefetch_factor=5 ,pin_memory=True)

    
    #Set maually the partiton type and hyperparameters here based on your needs, using the fedlab library
    dir_par =  CIFAR10Partitioner(trainset.targets,
                                 num_clients,
                                 balance=None,
                                 unbalance_sgm=0,
                                 partition="dirichlet",
                                 min_require_size= 1100,
                                 dir_alpha=da,
                                 )



    L = []
    for i in range(num_clients):
        l = len(dir_par.client_dict[i])
        L.append(l)


    csv_file = ".cifar10_hetero_dir_0.3_100clients.csv"
    partition_report(trainset.targets, dir_par.client_dict,
                    class_num=num_classes,
                    verbose=False, file=csv_file)


    hetero_dir_part_df = pd.read_csv(csv_file,header=1)

    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(num_classes)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)



    hetero_dir_part_df[col_names].iloc[:20].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(f"cifar10_hetero_dir_0.3_100clients.png", dpi=400)


    T_data = []
    for i in range(num_clients):
        
        td = copy.deepcopy(torch.utils.data.Subset(trainset, dir_par.client_dict[i]))
        T_data.append(td)


    train_loader = [torch.utils.data.DataLoader(x, batch_size=128, shuffle=True, num_workers = 8, prefetch_factor= 4 ,persistent_workers = True, pin_memory=True) for x in T_data]

    return train_loader, test_loader





def FashionMNISTPartition(num_clients, partition, dir_alpha, batch_size):

   

    num_classes = 10
    seed = random.randint(0, 1000000)

    trainset = datasets.FashionMNIST('./data', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor(), 
                                                        transforms.Normalize((0.5,), (0.5,))])

                          )



    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), 
                                                                                transforms.Normalize((0.5,), (0.5,))])
            ), batch_size=128, shuffle=True, num_workers = 12, pin_memory=True)


    #partition = "iid"
    
    dir_par = FMNISTPartitioner(trainset.targets, 
                                        num_clients=num_clients,
                                        partition=partition, 
                                        dir_alpha=dir_alpha,
                                        #min_require_size=600,
                                        seed=seed)

    csv_file = ".MNIST.csv"
    partition_report(trainset.targets, dir_par.client_dict,
                    class_num=num_classes,
                    verbose=False, file=csv_file)


    hetero_dir_part_df = pd.read_csv(csv_file,header=1)

    hetero_dir_part_df = hetero_dir_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(num_classes)]
    for col in col_names:
        hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)



    hetero_dir_part_df[col_names].iloc[:30].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(f"cifar10_hetero_dir_0.3_100clients.png", dpi=400)


    T_data = []
    for i in range(num_clients):
        
        td = copy.deepcopy(torch.utils.data.Subset(trainset, dir_par.client_dict[i]))
        T_data.append(td)


    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory=True) for x in T_data]

    return train_loader, test_loader