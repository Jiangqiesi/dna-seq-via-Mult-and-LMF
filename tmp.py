# import numpy as np
#
# a = np.array([[0.0010, 0.0018, 0.0115, 0.0055],
#               [-0.0014, 0.0015, 0.0070, 0.0081],
#               [0.0090, 0.0062, 0.0054, 0.0026]])
#
# b = np.array([[0., 1., 0., 0.],
#               [0., 0., 1., 0.],
#               [1., 0., 0., 0.]])
#
# base = np.array([[1., 1., 1., 1.],
#                  [1., 1., 1., 1.],
#                  [1., 1., 1., 1.]])
#
# c = (a - b)
# d = (base - np.abs(c))
# print(d)
# # 初始化
# e = np.zeros(len(c))
# for i in range(len(c)):
#     # 每项相加，然后减3，最后除4
#     f = ((d[i][0] + d[i][1] + d[i][2] + d[i][3]) - 2) / 4
#     print(f)
#     e[i] = f
#
# print(e)
import numpy as np

# 预测结果
a = np.array([[0.0010, 0.0018, 0.0115, 0.0055],
              [-0.0014, 0.0015, 0.0070, 0.0081],
              [0.0090, 0.0062, 0.0054, 0.0026]])

# 真实标签（one-hot 编码）
b = np.array([[0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [1., 0., 0., 0.]])
print(a * 100)
# 找到每一行的最大概率值索引
predicted_indices = np.argmax(a, axis=1)
true_indices = np.argmax(b, axis=1)

# 比较预测与真实标签
correct_predictions = (predicted_indices == true_indices)

print(predicted_indices)
print(correct_predictions)
# 计算置信度
confidence_scores = a[np.arange(a.shape[0]), predicted_indices]
mean_confidence = np.mean(confidence_scores[correct_predictions])  # 只计算正确预测的平均置信度
print(confidence_scores.shape)

print("置信度（每个样本）:", confidence_scores)
print("平均置信度（正确的预测）:", mean_confidence)
