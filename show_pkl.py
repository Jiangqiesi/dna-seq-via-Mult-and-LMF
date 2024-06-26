# show_pkl.py

import pickle

path = './X_q.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
file_path = './X_q.txt'

# f = open(path, 'rb')
# data = pickle.load(f)
#
# print(data)
# print(len(data))
with open(path, 'rb') as f:
    data = pickle.load(f)

# save as file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(repr(data))

# output len
print(len(data))

# # 输出结果：
# {'N_0000000356_00190': array(
#     [2.86, 2.27, 1.45, 1.1, 0., 0.65, 0.05, 0., 0.75, 1.65, 0.6, 0., 1.86, 0., 0.62, 0.25, 0.]),
#  'N_0000001939_00054': array([0.34, 2.09, 0., 2.04, 0.02, 0., 0., 1.22, 0., 0.93, 0.37, 0., 0.4, 0., 0., 0.22, 0.]),
#  'N_0000000437_00540': array(
#      [0., 0.19, 0.02, 0.8, 0.24, 1.46, 1.18, 0.37, 0., 0., 1.13, 3.37, 1.24, 0.73, 0.13, 1.83, 0.]),
#  'N_0000001507_00202': array(
#      [1.08, 1.23, 0., 1.83, 0.31, 1.08, 0.04, 0., 0.24, 1.31, 0., 0.25, 0.44, 0.6, 0.77, 0., 0.])}
# 4