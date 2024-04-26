import ast
import pickle

import numpy as np


# 函数用于读取FASTA格式的文件并返回一个序列字典
def read_fasta_file(fasta_file):
    sequences = {}
    with open(fasta_file, 'r') as file:
        sequence_id = None
        # sequence = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # if sequence_id is not None:
                #     sequences[sequence_id] = sequence
                sequence_id = int(line[1:])  # 移除 '>'
                # sequence = ''
            else:
                nested_list = ast.literal_eval(line)
                sequences[sequence_id] = nested_list
        # if sequence_id is not None:
        #     sequences[sequence_id] = sequence  # 添加最后一个序列
    return sequences


# 函数用于读取质量值文件并返回一个质量值字典
def read_qual_file(qual_file):
    quality_scores = {}
    with open(qual_file, 'r') as file:
        sequence_id = None
        # quality_score_list = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # if sequence_id is not None:
                #     quality_scores[sequence_id] = quality_score_list
                sequence_id = int(line[1:])  # 移除 '>'
                # quality_score_list = []
            else:
                # 清理方括号
                # cleaned_line = line.strip('[] \n')
                # # 假设质量值已经是以逗号分隔的数值
                # quality_scores_list = [int(q) for q in cleaned_line.split(',') if q.strip().isdigit()]
                # quality_score_list.append(quality_scores_list)
                nested_list = ast.literal_eval(line)
                quality_scores[sequence_id] = nested_list
        # if sequence_id is not None:
        #     quality_scores[sequence_id] = quality_score_list  # 添加最后一组质量值
    return quality_scores


# 读取原始序列
# ori_seqs = read_fasta_file('./data/seq260all_ori_seqs.fasta')
# print(ori_seqs[0])

# 读取高拷贝序列
# high_copy_seqs = read_fasta_file('./data/seq260all_seqs.fasta')

# 读取质量值
quals = read_qual_file('./data/seq260all_quas.fasta')
print(quals[0][0])

# print(ori_seqs, high_copy_seqs, quals)
# print(list(ori_seqs.items())[0])
# print(list(high_copy_seqs.items())[0])
# print(list(quals.items())[0])

#
# # 数据预处理和整合的示例
# # copy自read_data.py
# # 用于DNA序列的one-hot编码
# def one_hot_encode_dna(sequence):
#     # 定义碱基到索引的映射
#     base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
#     # 创建与序列长度相等的one-hot编码矩阵，每个碱基对应一个长度为4的向量
#     one_hot_sequence = np.zeros((len(sequence), 4))
#     for i, base in enumerate(sequence):
#         if base in base_to_index:
#             one_hot_sequence[i, base_to_index[base]] = 1
#     return one_hot_sequence
#
#
# # 用于质量值的归一化
# def normalize_quality_scores(quality_scores):
#     # 将质量值归一化到0-1的范围
#     normalized_quality_scores = np.array(quality_scores) / 50
#     return normalized_quality_scores
#
#
# # 根据给定的数据结构读取并整合数据
# def integrate_data(high_copy_seqs, quals):
#     X_c = {}  # 用于编码后的DNA序列
#     X_q = {}  # 用于归一化的质量分数
#     flag = False
#
#     # 遍历高拷贝序列，将它们和对应的质量分数整合到一起
#     for seq_id in high_copy_seqs.keys():
#
#         # if not flag:
#         #     print("seq list type:", type(high_copy_seqs[seq_id]))
#         #     flag = True
#         seq_list = ast.literal_eval(high_copy_seqs[seq_id])
#         # for seq in seq_list:
#         #     if not flag:
#         #         print("seq:", seq)
#         #         flag = True
#
#         encoded_seqs = [one_hot_encode_dna(seq) for seq in seq_list]
#         # 对每个质量分数进行归一化
#         normalized_quals = [normalize_quality_scores(q) for q in quals[seq_id]]
#
#         # 用列表存储每个序列ID下的多个序列和质量分数
#         X_c[seq_id] = []
#         X_q[seq_id] = []
#
#         for encoded_seq, qual in zip(encoded_seqs, normalized_quals):
#             if not flag:
#                 print(f"Seq ID: {seq_id}, Encoded Length: {len(encoded_seq)}, Qual Length: {len(qual)}")  # 调试输出
#                 flag = True
#             # if len(encoded_seq) == len(qual):
#             X_c[seq_id].append(encoded_seq)
#             X_q[seq_id].append(qual)
#             # else:
#             #     print(f"Mismatch found in Seq ID: {seq_id}")  # 发现长度不匹配时打印
#     return X_c, X_q
#
#
# # 预处理并整合数据
# X_c, X_q = integrate_data(high_copy_seqs, quals)
# # 对每个序列使用one_hot_encode_dna函数进行编码
# # encoded_ori_seqs = {seq_id: [one_hot_encode_dna(seq) for seq in seqs] for seq_id, seqs in ori_seqs.items()}
# print(list(X_c.items())[0])
#
#
# # 保存为pickle文件
# def save_to_pickle(data, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(data, file)
#
#
# save_to_pickle(X_c, 'X_c.pkl')
# save_to_pickle(X_q, 'X_q.pkl')
# # save_to_pickle(encoded_ori_seqs, 'encoded_ori_seqs.pkl')
