import numpy as np


# 用于DNA序列的one-hot编码
def one_hot_encode_dna(sequence):
    # 定义碱基到索引的映射
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # 创建与序列长度相等的one-hot编码矩阵，每个碱基对应一个长度为4的向量
    one_hot_sequence = np.zeros((len(sequence), 4))
    for i, base in enumerate(sequence):
        if base in base_to_index:
            one_hot_sequence[i, base_to_index[base]] = 1
    return one_hot_sequence


# 用于质量值的归一化
def normalize_quality_scores(quality_scores):
    # 将质量值归一化到0-1的范围
    normalized_quality_scores = np.array(quality_scores) / 50
    return normalized_quality_scores


# 根据给定的数据结构读取并整合数据
def integrate_data(high_copy_seqs, quals):
    X_c = {}  # 用于编码后的DNA序列
    X_q = {}  # 用于归一化的质量分数

    # 遍历高拷贝序列，将它们和对应的质量分数整合到一起
    for seq_id in high_copy_seqs.keys():
        # 对每个高拷贝序列进行one-hot编码
        encoded_seqs = [one_hot_encode_dna(seq) for seq in high_copy_seqs[seq_id]]
        # 对每个质量分数进行归一化
        normalized_quals = [normalize_quality_scores(q) for q in quals[seq_id]]

        # 用列表存储每个序列ID下的多个序列和质量分数
        X_c[seq_id] = []
        X_q[seq_id] = []

        for encoded_seq, qual in zip(encoded_seqs, normalized_quals):
            if len(encoded_seq) == len(qual):
                X_c[seq_id].append(encoded_seq)
                X_q[seq_id].append(qual)

    return X_c, X_q


# 示例数据
ori_seqs_sample = {
    '0': ['CAGTTA']
}

high_copy_seqs_sample = {
    '0': ['CAGTA', 'CAGT'],
    '1': ['CCGATC']
}

quals_sample = {
    '0': [[32, 34, 30, 28, 35], [30, 32, 33, 28]],
    '1': [[35, 30, 25, 28, 32, 31]]
}

# 预处理并整合数据
X_c_sample, X_q_sample = integrate_data(high_copy_seqs_sample, quals_sample)
# 对每个序列使用one_hot_encode_dna函数进行编码
encoded_seqs = {seq_id: [one_hot_encode_dna(seq) for seq in seqs] for seq_id, seqs in ori_seqs_sample.items()}

# 打印出整合后的数据示例
print("Encoded DNA Sequences (X_c):", X_c_sample)
print("Normalized Quality Scores (X_q):", X_q_sample)
print("Origin Sequences:", encoded_seqs)
