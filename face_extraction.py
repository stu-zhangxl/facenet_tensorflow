import numpy as np
from model.model import build_model


# 图片预处理
# 高斯归一化
def pre_process(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y


#---------------------------------#
#   l2标准化
#---------------------------------#
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output
#---------------------------------#
#   计算特征值
#---------------------------------#


# 定义facenet人脸检测处理类
class Face_Extraction():
    def __init__(self):
        self.facenet = build_model()
        model_path = './weights/model.01-0.2455.h5'
        self.facenet.load_weights(model_path)
        self.img_encodings = []

    def calc_vec(self, img):
        # face_img = pre_process(img)
        pre = self.facenet.predict(img)
        pre = l2_normalize(np.concatenate(pre))
        pre = np.reshape(pre, [128])
        return pre

    def batch_extract(self, imgs, rectangles):
        for img, rectangle in zip(imgs, rectangles):
            self.img_encodings.append(self.calc_vec(img))
        return self.img_encodings




