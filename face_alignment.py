import cv2
import numpy as np
import math

#-------------------------------------#
#   人脸对齐
#-------------------------------------#


def Alignment(img, landmark):
    # 计算两点间x方向和y方向的差
    x = landmark[0, 0] - landmark[1, 0]
    y = landmark[0, 1] - landmark[1, 1]

    if x == 0:
        angle = 0
    else:
        # 计算旋转角度
        angle = math.atan(y/x)*180/math.pi
    # 获取图像中心
    center = (img.shape[1]//2, img.shape[0]//2)

    # 计算旋转矩阵
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 旋转变换
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    # 遍历所有关键点，对所有关键点进行同样的旋转变换
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0]*landmark[i, 0]+RotationMatrix[0, 1]*landmark[i, 1]+RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0]*landmark[i, 0]+RotationMatrix[1, 1]*landmark[i, 1]+RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark