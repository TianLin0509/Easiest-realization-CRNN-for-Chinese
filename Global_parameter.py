import os
# 产生字典
def get_letters():
    with open('dictionary.txt', encoding='utf8') as f1:
        str_word = f1.read()
        list_word = list(str_word)
        list_word = [y for y in list_word if y not in '\', ']
        list_word = list_word[1:-1]
    return list_word

letters = get_letters()
# 包括了空格
num_classes = len(letters) + 1

# 设置统一图片的尺寸（不符合的图片会被按比例裁剪）
img_w, img_h, img_c = 640, 64, 3

# 超参数设置
batch_size = 1
downsample_factor = 4
# 最长的标签的长度 （图片最多包含的字数)
max_text_len = 30
cur_index = 0

# 训练集， 测试集， 模型保存路径
Train_file = "./train_photo/"
Pred_file="./pred_photo/"
model_path = '1.hdf5'
train_files = os.listdir(Train_file)
n = len(train_files)    # 训练图片总数 （样本总数)
indexes = list(range(n))

