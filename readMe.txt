关于工程文件夹中各文件的说明：
1.model文件夹：定义了各个模块需要用到的网络模型
2.weights文件夹：包含了已训练好了的mtcnn和facenet的网络权重文件
3.tmp文件夹：程序运行过程中产生的临时数据的存放路径
4.face_database文件夹：后台人脸数据集
5.test_images文件夹：待识别的人脸照片存放路径
6.face_alignment.py：定义了人脸图像对齐函数
7.get_face_database.py：定义了加载后台人脸照片数据的类
8.face_detection.py：主要定义了人脸检测的类
9.face_extraction.py：主要定义了人脸图像特征提取的类
10.face_match.py：主要定义了用于人脸识别（人脸特征匹配）的类
11.get_faces.py：定义了从读取照片到获取人脸图像数据的函数
12.main.py：主脚本
