import face_detection as fd
import cv2
import numpy as np
import face_alignment as fa
import os


def get_faces(image=None, tmp=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 调换颜色通道的顺序
    image_copy = image.copy()

    face_detector = fd.Face_Detection()                 # 人脸检测器
    rectangles = face_detector.detectFace(image_copy)        # 人脸检测
    rectangles = fd.rect2square(np.array(rectangles))        # 将检测结果转为正方形
    faces = []      # 用于保存截取到的人脸图像

    for idx, rectangle in enumerate(rectangles):
        bbox = rectangle[0:4]           # 人脸框的左上角和右下角坐标（注意照片坐标系横轴向右为正，纵轴向下为正）
        points = rectangle[-10:]        # 人脸五个关键位置（两只眼睛、一个鼻子、两个嘴角）

        cv2.rectangle(image_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0))   # 绘制人脸框
        # 画出人脸五个关键位置（两只眼睛、一个鼻子、两个嘴角）
        for i in range(5):
            cv2.circle(image_copy, (int(points[i*2]), int(points[i*2+1])), 4, (0, 0, 255), 5)

        detect_image = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)    # 调整颜色通道顺序
        cv2.imwrite(os.path.join(tmp, 'detect{}.jpg'.format(idx)), detect_image)                   # 保存照片

        # 截取人脸部分的图像信息
        crop_image = image[int(rectangle[1]): int(rectangle[3]), int(rectangle[0]): int(rectangle[2]), :]
        crop_image = cv2.resize(crop_image, (160, 160))     # 压缩成指定大小

        # 人脸对齐,矩阵仿射变换：将人脸摆正
        landmark = (np.reshape(rectangle[-10:], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
                rectangle[3] - rectangle[1])     # 记下他们的landmark
        alignment_image, new_landmark = fa.Alignment(crop_image, landmark)         # 执行仿射变换：对齐人脸

        new_crop_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)               # 调整颜色通道顺序
        new_alignment_image = cv2.cvtColor(alignment_image, cv2.COLOR_RGB2BGR)     # 调整颜色通道顺序

        cv2.imwrite(os.path.join(tmp, 'crop_image{}.jpg'.format(idx)), new_crop_image)                # 保存照片
        cv2.imwrite(os.path.join(tmp, 'alignment_image{}.jpg'.format(idx)), new_alignment_image)      # 保存照片

        alignment_image = np.expand_dims(alignment_image, 0)             # 拓展一个维度
        faces.append(alignment_image)
    return faces, rectangles


