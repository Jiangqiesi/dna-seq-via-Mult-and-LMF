import pickle

import numpy as np


# 函数用于读取FASTA格式的文件并返回一个序列字典
# 在read_fasta_file函数中添加打印语句
def read_fasta_file(fasta_file):
    sequences = {}
    with open(fasta_file, 'r') as file:
        sequence_id = None
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id is not None:
                    sequences[sequence_id] = sequence
                    print(f"Complete sequence for ID {sequence_id}: {sequence}")  # 打印完整序列
                sequence_id = line[1:]
                sequence = ''
            else:
                sequence += line
        if sequence_id is not None:
            sequences[sequence_id] = sequence
            print(f"Complete sequence for ID {sequence_id}: {sequence}")  # 打印最后一个完整序列
    return sequences


# 读取原始序列
ori_seqs = read_fasta_file('./data/seq260all_ori_seqs_tmp.fasta')
print(list(ori_seqs.items()))


# 用于DNA序列的one-hot编码
def one_hot_encode_dna(sequence):
    # print(sequence)  # 输出调试
    # 定义碱基到索引的映射
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # 创建与序列长度相等的one-hot编码矩阵，每个碱基对应一个长度为4的向量
    one_hot_sequence = np.zeros((len(sequence), 4))
    for i, base in enumerate(sequence):
        if base in base_to_index:
            one_hot_sequence[i, base_to_index[base]] = 1
    return one_hot_sequence


# 对每个序列使用one_hot_encode_dna函数进行编码
# encoded_ori_seqs = {seq_id: [one_hot_encode_dna(seq) for seq in seqs] for seq_id, seqs in ori_seqs.items()}
# Assuming ori_seqs is a dictionary with sequence IDs as keys and complete DNA sequences as values
encoded_ori_seqs = {}
for seq_id, seq in ori_seqs.items():
    print(f"Encoding sequence for ID {seq_id}: {seq}")  # Debug print to check what is being encoded
    encoded_ori_seqs[seq_id] = one_hot_encode_dna(seq)
print(list(encoded_ori_seqs.items()))
