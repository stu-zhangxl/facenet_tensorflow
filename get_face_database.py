import os
import face_detection as fd
import face_extraction as fe
import face_alignment as fa
from model.model import build_model
import cv2
import numpy as np


# 定义facenet人脸检测处理类
class Face_Database:
    def __init__(self, path):
        self.face_detection_model = fd.Face_Detection()
        self.path = path
        self.face_list = os.listdir(path)
        self.face_extraction_model = fe.Face_Extraction()
        self.known_face_encodings = []
        self.known_face_names = []

    def build_database(self):
        # 遍历人脸数据库
        for face in self.face_list:
            name = face.split(".")[0]
            # 读取图像
            img = cv2.imread(self.path + face)
            # 转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape
            # 检测人脸
            rectangles = self.face_detection_model.detectFace(img)
            # 转化成正方形
            if len(rectangles) == 0:
                continue
            rectangles = fd.rect2square(np.array(rectangles))

            pick = []
            # 遍历人脸框，防止人脸框越界
            for i in range(len(rectangles)):
                x1 = int(max(0, rectangles[i][0]))
                y1 = int(max(0, rectangles[i][1]))
                x2 = int(min(width, rectangles[i][2]))
                y2 = int(min(height, rectangles[i][3]))
                if x2 > x1 and y2 > y1:
                    rectangles[i][0] = x1
                    rectangles[i][1] = y1
                    rectangles[i][2] = x2
                    rectangles[i][3] = y2
            # facenet要传入一个160x160的图片
            rectangle = rectangles[0]
            # 记下landmark
            landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                    rectangle[3] - rectangle[1]) * 160
            # 裁剪人脸图像
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img, (160, 160))
            # 对齐人脸
            new_img, _ = fa.Alignment(crop_img, landmark)
            # 扩展一个维度
            new_img = np.expand_dims(new_img, 0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = self.face_extraction_model.calc_vec(new_img)
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)


