from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model

# -----------------------------#
#   Pnet网络粗略获取人脸框
#   输出bbox位置和是否有人脸
# -----------------------------#


def create_Pnet(weight_path):
    # 定义Pnet网络输入
    input = Input(shape=[None, None, 3])

    # 3x3卷积+prelu+最大池化
    # h,w,3 -> h/2,w/2,10
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 3x3卷积+prelu
    # h/2,w/2,10 -> h/2,w/2,16
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    # 3x3卷积+prelu
    # h/2,w/2,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # 1x1卷积，人脸分类
    # h/2, w/2, 2
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)

    # 1x1卷积，人脸框定位
    # h/2, w/2, 4
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])

    # 加载权重文件
    model.load_weights(weight_path, by_name=True)
    return model

# -----------------------------#
#   mtcnn的第二段Rnet
#   精修框
# -----------------------------#
def create_Rnet(weight_path):
    # 定义Rnet网络输入
    input = Input(shape=[24, 24, 3])

    # 3x3卷积+prelu+最大池化
    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 3x3卷积+prelu+最大池化
    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 3x3卷积+prelu
    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 特征拉直
    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 全连接层+prelu
    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    # 全连接层+softmax人脸分类，全连接层人脸定位
    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# -----------------------------#
#   mtcnn的第三段Onet
#   精修框并获得五个点
# -----------------------------#
def create_Onet(weight_path):
    # 定义Onet网络输入
    input = Input(shape=[48, 48, 3])

    # 3x3卷积+prelu+最大池化
    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 3x3卷积+prelu+最大池化
    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 3x3卷积+prelu+最大池化
    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 3x3卷积+prelu
    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # 特征维度更换顺序
    # 3,3,128 -> 128,3,3
    x = Permute((3, 2, 1))(x)

    # 特征拉直+全连接层+prelu
    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 全连接+softmax人脸分类 256 -> 2
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    # 全连接+人脸定位 256 -> 4
    bbox_regress = Dense(4, name='conv6-2')(x)
    # 全连接层回归人脸关键点 256 -> 10
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model