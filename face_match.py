import os
import face_detection as fd
import numpy as np
import cv2



#---------------------------------#
#   计算人脸距离
#---------------------------------#
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


#---------------------------------#
#   比较人脸
#---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance)


# 定义facenet人脸检测处理类
class Face_Match:
    def __init__(self, img_encodings, face_encodings, face_names):
        self.img_encodings = img_encodings
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.img_similarities = []

    def validation(self, img_vali_encodings):
        img_distances = face_distance(np.array(self.img_encodings), np.array(img_vali_encodings))
        self.img_similarities = 1 - img_distances


    def recognize(self, tolerance=0.1):
        match_names = []
        for img_encoding in self.img_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            match_face_encodings = compare_faces(self.known_face_encodings, img_encoding, tolerance)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = face_distance(self.known_face_encodings, img_encoding)
            # print(face_distances)
            # 取出这个最近人脸的下标
            best_match_index = np.argmin(face_distances)
            if match_face_encodings[best_match_index]:
                name = self.known_face_names[best_match_index]
            match_names.append(name)
        return match_names

    def draw_rectangle(self, img, rectangles, names):

        rectangles = rectangles[:, 0:4]
        # -----------------------------------------------#
        #   画框
        # -----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, names):
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, name, (int(left), int(bottom) - 15), font, 0.75, (255, 255, 255), 2)
        return img


    def print_similarity(self):
        print(self.img_similarities)




