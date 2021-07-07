import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import sampler,Subset
import numpy as np
import random


def build_dataset(arg):
    print('>>> Preparing data..')
    Datalist=[]
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,), (0.3081,))
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                            transform=transform_train)
    random_sampler = sampler.RandomSampler(data_source=trainset)  # 打乱数据的索引
    batch_sampler = sampler.BatchSampler(random_sampler, int(len(trainset) / arg.K), True)  # 生成一定数量的索引的集合
    for i in batch_sampler:
        Datalist.append(
            dataloader.DataLoader(Subset(trainset, i), batch_size=arg.batch_size, shuffle=True))  # 按照索引生成子数据集

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                           transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=arg.test_batch_size, shuffle=False)


    return Datalist, test_loader
def com_para(paralist,arg):
    com={}
    for key in paralist[0]:
        com[key]=paralist[0][key].float()
        paralist[0][key]=None
    for i in range(1,len(paralist)):
        for key in com:
            com[key]+=paralist[i][key].float()

            paralist[i][key] = None
    client_num=arg.K*arg.C
    for key in com:
        com[key]/=client_num
    torch.cuda.empty_cache()
    return com
def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
