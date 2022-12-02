from get_faces import get_faces
import cv2
import matplotlib.pyplot as plt
import face_extraction as fe
import get_face_database as fdb
import face_match as fm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # key

image = cv2.imread('test_images/image2.jpg')         # 读取照片
faces, rectangles = get_faces(image, 'tmp')          # 获取照片中的人脸图像，并将其对齐

# 获取人脸图像的特征向量
face_feature_extractor = fe.Face_Extraction()
face_features = face_feature_extractor.batch_extract(faces, rectangles)    # 获取人脸图像的特征向量

# 加载后台数据库人数照片
database_path = 'face_database/'                       # 后台人脸数据库路径
face_database = fdb.Face_Database(database_path)
face_database.build_database()
face_names = face_database.known_face_names            # 后台数据库人名信息
face_encodings = face_database.known_face_encodings    # 后台数据库中人脸特征向量

# 人脸识别
face_match_model = fm.Face_Match(face_features, face_encodings, face_names)     # 计算向量之间的距离
names = face_match_model.recognize()           # 执行识别操作
print('照片中的人是：', names)

recognize_image = face_match_model.draw_rectangle(image, rectangles, names)   # 将识别结果标注至照片中
cv2.imwrite('tmp/recognize.jpg', recognize_image)                             # 保存照片

