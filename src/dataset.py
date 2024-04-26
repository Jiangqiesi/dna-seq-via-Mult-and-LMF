import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META


class multiseqs_datasets(Dataset):
    def __init__(self, dataset_path='./data/', data1='encoded_ori_seqs.pkl', data2='X_c.pkl', data3='X_q.pkl', split_type='train'):
        super(multiseqs_datasets, self).__init__()
        data1 = os.path.join(dataset_path, data1)
        dataset1 = pickle.load(open(data1, 'rb'))
        data2 = os.path.join(dataset_path, data2)
        dataset2 = pickle.load(open(data2, 'rb'))
        data3 = os.path.join(dataset_path, data3)
        dataset3 = pickle.load(open(data3, 'rb'))

        # self.seqs = torch.tensor(dataset2.astype(np.float32)).cpu().detach()
        self.seqs = {}
        for key, value in dataset2.items():
            try:
                converted = torch.tensor(value.astype(np.float32)).cpu().detach()
                self.seqs[key] = converted
            except AttributeError:
                print(f"Warning: Item {key} skipped as it does not support astype operation")

        # self.quas = torch.tensor(dataset3.astype(np.float32)).cpu().detach()
        # self.ori_seqs = torch.tensor(dataset1.astype(np.float32)).cpu().detach()
        self.quas = {}
        for key, value in dataset3.items():
            try:
                converted = torch.tensor(value.astype(np.float32)).cpu().detach()
                self.quas[key] = converted
            except AttributeError:
                print(f"Warning: Item {key} skipped as it does not support astype operation")

        self.ori_seqs = {}
        for key, value in dataset1.items():
            try:
                converted = torch.tensor(value.astype(np.float32)).cpu().detach()
                self.ori_seqs[key] = converted
            except AttributeError:
                print(f"Warning: Item {key} skipped as it does not support astype operation")

        self.data1 = data1
        self.data2 = data2
        self.data3 = data3

        self.n_modalities = 2

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.seqs.shape[1], self.quas.shape[1]

    def get_dim(self):
        return self.seqs.shape[2], self.quas.shape[2]

    def __len__(self):
        return self.ori_seqs.shape[0]

    def __getitem__(self, idx):
        x = (idx, self.seqs[idx, :], self.quas[idx, :, :])
        y = self.ori_seqs[idx, :]
        return x, y


# 测试代码
print("test start")
data_path = '/Users/jiangqiesi/Documents/code/PycharmProjects/Multimodal-Transformer/data/'
save_data_file = data_path + 'tmp.pkl'
data = multiseqs_datasets(data_path)
torch.save(data, save_data_file)
