import numpy as np


def one_hot_encode_dna_sequence(sequence):
    # 定义碱基的种类数
    base_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    num_bases = len(base_dict)

    # 创建空的编码矩阵
    encoding = np.zeros((len(sequence), num_bases))

    # 进行One-Hot编码
    for i, base in enumerate(sequence):
        encoding[i, base_dict[base]] = 1

    return encoding


# 获取用户输入的DNA序列
input_sequence = input()
dna_sequences = input_sequence.split(',')

# 进行One-Hot编码
encoded_sequences = []
for seq in dna_sequences:
    encoded_seq = one_hot_encode_dna_sequence(seq)
    encoded_sequences.append(encoded_seq)

# 输出编码结果
for i, seq in enumerate(encoded_sequences):
    print(f"DNA sequence {i+1} encoding:")
    print(seq)
    print()