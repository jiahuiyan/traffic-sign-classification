import numpy as np
import os
import cv2
import random

raw_data = []
normalized_data = []

raw_data_path = './raw_training_2'
data_path = './training_data'
# test_data_path = './test_data'

if not os.path.exists(raw_data_path):
    print('missing raw dataset!')

if not os.path.exists(data_path):
    os.makedirs(data_path)

# if not os.path.exists(test_data_path):
#     os.makedirs(test_data_path)
j=0
for dirname in os.listdir(raw_data_path):
    i = 0
    for filename in os.listdir(raw_data_path+'/'+dirname):
        if filename.endswith('ppm'):
            img = cv2.imread(raw_data_path+'/'+dirname+'/'+filename)
            
            # 对图片进行resize处理
            img = cv2.resize(img,(32,32))
            
            # 将RGB 图片转换成灰度
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            print(data_path+'/'+dirname+'/'+str(i)+'.jpg')
            if not os.path.exists(data_path+'/'+dirname):
                os.makedirs(data_path+'/'+dirname)
            cv2.imwrite(data_path+'/'+dirname+'/'+str(i)+'.jpg',img)
            i = i+1
    j=j+i

print('totol number: '+str(j))


# 定义随机相司变换
def applyRandSimilarityTran(image, n):
    output_images = np.zeros((n,32,32))# 生成n个32X32的矩阵
    # 对每张输入的图片进行处理
    for i in range(n):
        angle = random.uniform(-15, 15) # 从-15到15中间随机取一个数

        s = random.uniform(0.7, 1.3)    # scale？？？将图片缩放

        rows,cols = image.shape[0:2]
        image_center = (rows/2.0, cols/2.0)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
        M_rot = np.vstack([rot_mat,[0,0,1]])

        tx = random.uniform(-2, 2)      # translation along x axis
        ty = random.uniform(-2, 2)      # translation along y axis
        M_tran = np.float32([[1,0,tx],[0,1,ty],[0,0,1]])

        M = np.matrix(M_tran) * np.matrix(M_rot)

        M = np.float32(M[:2][:]) # similarity transform

        tmp = cv2.warpAffine(image, M, (cols, rows))    
        output_images[i][:][:] = tmp
        
        cv2.equalizeHist(image, image)
        
    return output_images