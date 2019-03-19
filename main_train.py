from keras.callbacks import *
from Model import get_Model
from Samples_generation import *
K.set_learning_phase(0)

# 获取模型
model = get_Model(training=True)
# 载入模型权重
try:
    model.load_weights(model_path)
    print("...Previous weight data...")
except:
    print("...New weight data...")


# 将损失最低的模型保存在名为1.hdf5的文件中
checkpoint = ModelCheckpoint(filepath=model_path, monitor='loss', verbose=1, mode='min', period=1)
# 损失函数实际上已经在model内部算好了并作为y_pred输出了，所以这里只需要写这样一个loss就可以了
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
model.fit_generator(generator=next_batch(),
                    steps_per_epoch=int(n / batch_size),
                    callbacks=[checkpoint],
                    epochs=3000)
