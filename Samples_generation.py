import cv2, itertools
import os, random
import numpy as np
from Global_parameter import *
import linecache


# 用于从text文件中读取标签
def get_line_context(number, file_path='./label.txt'):
    # 读取txt文件中的第number + 1行
    text = linecache.getline(file_path, number + 1).strip()
    # print(text)
    return text


# cv2读取图片，并转化为numpy数组表示的样本
def gen_img(number):
    # 图片路径
    img_path = Train_file + train_files[number]
    # print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_h, img_w))
    img = np.rot90(img)
    img = img.astype(np.float32)
    img = (img / 255.0) * 2.0 - 1.0
    return img


# 产生一组 numpy数组表示的图片 和 对应的文字标签
def next_sample():
    global cur_index
    cur_index += 1
    if cur_index >= n:
        cur_index = 0
        random.shuffle(indexes)
    text = get_line_context(indexes[cur_index])
    img = gen_img(indexes[cur_index])
    return img, text


def next_batch():  ## batch size만큼 가져오기
    while True:
        X_data = np.ones([batch_size, img_w, img_h, img_c])
        Y_data = np.ones([batch_size, max_text_len])
        input_length = np.ones((batch_size, 1)) * 30   # 这个参数的设置我也不太清楚 希望有人可以解释
        label_length = np.zeros((batch_size, 1))

        for i in range(batch_size):
            img, text = next_sample()
            # 因为我做的是竖着写的中文检测，所以要进行转置。 横着的则可以注释掉下一行
            img = np.transpose(img, [1, 0, 2])
            # img = np.expand_dims(img, -1)
            X_data[i] = img
            new_label = text_to_labels(text)
            Y_data[i, :len(new_label)] = new_label
            label_length[i] = len(text)

        inputs = {
            'the_input': X_data,
            'the_labels': Y_data,
            'input_length': input_length,
            'label_length': label_length
        }
        # 因为 y_true并没有用， 所以随便生成一个满足条件的outputs即可
        outputs = {'ctc': np.zeros([batch_size])}
        yield (inputs, outputs)


# 把文字转换为数字，作为标签， 每个文字被转换为字典letters中一一对应的index值
def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


# 把网络输出的数字标签翻译回letters中一一对应的中文
def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr

# 如果需要先对一张大的图片进行裁剪，获取文字部分
def cut_img(ab_path, position, count):
    left, right, bottom, top = position
    img = cv2.imread(ab_path)
    print(img.shape)
    cropped = img[top:bottom, left:right]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("./new_set/" + str(count) +".jpg", cropped)
    return cropped



