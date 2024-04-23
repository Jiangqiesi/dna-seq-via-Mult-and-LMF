import numpy as np

# 假设这是从文件读取的序列数据，我们将使用简短的序列进行演示
ori_seqs = {
    '0': 'CAGTACTGATGATGCTTTAT',
    '1': 'CCGATCATATGGTCCGGCT'
}

# 假设这是从文件读取的高拷贝序列，包含多个高拷贝变体
high_copy_seqs = {
    '0': ['CAGTA', 'CAGT', 'CAGTACTGAT'],
    '1': ['CCGAT', 'CCGATCATAT']
}

# 假设这是从文件读取的高拷贝序列对应的质量值
quals = {
    '0': [[32, 34, 35, 36, 14], [34, 36, 35, 17], [30, 32, 33, 36, 38, 39, 40, 42, 20, 22]],
    '1': [[33, 37, 38, 40, 15], [32, 34, 36, 39, 35, 37, 16, 18, 19, 21]]
}

# 定义DNA序列的one-hot编码映射
base_to_onehot = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
}


# one-hot编码函数
def one_hot_encode(sequence):
    return np.array([base_to_onehot[base] for base in sequence])


# 归一化质量值的函数
def normalize_quality(quality_scores):
    return np.array(quality_scores) / 50  # 由于质量值范围是1-50，所以用50归一化


# 初始化X_c和X_q列表，它们将包含所有序列和质量值的编码
X_c = []
X_q = []

# 处理序列和质量分数
for seq_id, seq_list in high_copy_seqs.items():
    for i, seq in enumerate(seq_list):
        # one-hot编码序列
        encoded_seq = one_hot_encode(seq)
        X_c.append(encoded_seq)

        # 归一化质量值
        normalized_qual = normalize_quality(quals[seq_id][i])
        X_q.append(normalized_qual)

# 现在我们把列表转换成Numpy数组
X_c = np.array(X_c, dtype=object)  # 使用dtype=object以容纳不同长度的序列
X_q = np.array(X_q, dtype=object)

# 输出X_c和X_q的形状来验证
print(X_c.shape, X_q.shape, X_c[0], X_q[0])
