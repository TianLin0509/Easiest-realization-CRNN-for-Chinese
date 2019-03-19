from Model import get_Model
from keras import backend as K
from Samples_generation import *

K.set_learning_phase(0)

# 获得模型
model = get_Model(training=False)

try:
    model.load_weights(model_path)
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")


def pre_label(img):
    img = cv2.resize(img, (img_h, img_w))
    img = np.rot90(img)
    img = img.astype(np.float32)
    img = np.transpose(img, [1, 0, 2])
    img_pred = (img / 255.0) * 2.0 - 1.0
    img_pred = np.expand_dims(img_pred, axis=0)
    net_out_value = model.predict(img_pred)
    pred_texts = decode_label(net_out_value)
    print('Predicted: %s' % (pred_texts))
    return pred_texts

if __name__ == '__main__':
    img_files = os.listdir(Pred_file)
    for img_path in img_files:
        img = cv2.imread(Pred_file + img_path)
        pre_label(img)






