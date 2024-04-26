import pickle

# 从.pkl文件中加载数据
with open('./data/X_c.pkl', 'rb') as f:
    data = pickle.load(f)

# 打印数据类型
print(type(data))


# 如果数据是嵌套的，可以使用递归函数进行遍历和打印
def print_data_type(data, level=0):
    flag_key = 0
    flag_list = 0
    if level > 2:
        return
    if isinstance(data, dict):
        print("  " * level + "Dictionary:")
        for key, value in data.items():
            flag_key += 1
            print("  " * (level+1) + f"Key: {key}")
            print_data_type(value, level+1)
            if flag_key >= 5:
                break
    elif isinstance(data, list):
        print("  " * level + "List:")
        for item in data:
            flag_list += 1
            print_data_type(item, level+1)
            if flag_list >= 5:
                break
    else:
        print("  " * level + f"Type: {type(data)}")


print_data_type(data)
