import numpy as np
import random
import math
import cv2
import os
import sys
import win_unicode_console
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt

win_unicode_console.enable()

data_path = './data'

imgs = [] # 存储所有读入的图像训练数据
labs = [] # 存储所有读入的图像训练标签
imgs_test = [] # 存储所有读取的图像测试数据
labs_test = [] # 存储所有读取的图像测试标签
size = 32 # 图像的大小

# 准备画图
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Traffic Sign Classification')
plt.xlabel('Epoch number')
plt.ylabel('Validation Accuracy')
fig_x = []
fig_y = []

def getPaddingSize(img):
    h, w = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

# 读取数据
def readTrainData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))
            # print(filename)
            imgs.append(img)
            temp = int(path.split('/')[-1])
            labs.append(temp)

# 读取源数据
print()
print('Reading data')
for dirname in os.listdir(data_path):

    readTrainData(data_path+'/'+dirname)
    
print("train_imgs = " + str(len(imgs)))
print("train_labs = " + str(len(labs)))

# 将两个列表转换成array
imgs = np.array(imgs)
labs = np.array(labs)

# 随机划分测试集valid集
train_x,valid_x,train_y,valid_y = train_test_split(imgs, labs, test_size=0.2, random_state=random.randint(0,100))
train_x,test_x,train_y,test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=random.randint(0,100))

# 将三个集合都改成需要的形状（dimention）
train_x = train_x.reshape(train_x.shape[0], size, size, 1)
test_x = test_x.reshape(test_x.shape[0], size, size, 1)
valid_x = valid_x.reshape(valid_x.shape[0], size, size, 1)

# 将数据标准化为[-1,1]
train_x = train_x.astype('float32')/255.0   
test_x = test_x.astype('float32')/255.0
valid_x = valid_x.astype('float32')/255.0

# 学习速率
rate = 0.001
# Epochs 大小
EPOCHS = 50
# batch 大小
BATCH_SIZE = 128

# def dnn(x):
#     mu = 0
#     sigma = 0.1
#     x = flatten(x)

#     l1_W  = tf.Variable(tf.truncated_normal(shape=(1024,10), mean = mu, stddev = sigma),name="l1_W")
#     l1_b  = tf.Variable(tf.zeros(10),name="l1_b")
#     l1 = tf.matmul(x, l1_W) + l1_b
#     l1 = tf.nn.relu(l1)

#     output_W  = tf.Variable(tf.truncated_normal(shape=(10, 43), mean = mu, stddev = sigma),name="fc3_W")
#     output_b  = tf.Variable(tf.zeros(43),name="fc3_b")
#     logits = tf.matmul(l1, output_W) + output_b

#     return logits

def cnn(x):    
    mu = 0
    sigma = 0.1
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name="conv1_W")
    conv1_b = tf.Variable(tf.zeros(6),name="conv1_b")
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # SOLUTION: Activation.
    conv1 = tf.nn.selu(conv1)
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma),name="conv2_W")
    conv2_b = tf.Variable(tf.zeros(16),name="conv2_b")
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # SOLUTION: Activation.
    conv2 = tf.nn.selu(conv2)
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    # # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    # fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma), name="fc1_W")
    # fc1_b = tf.Variable(tf.zeros(120),name="fc1_b")
    # fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    # # SOLUTION: Activation.
    # fc1    = tf.nn.relu(fc1)
    # # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    # fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma),name="fc2_W")
    # fc2_b  = tf.Variable(tf.zeros(84),name="fc2_b")
    # fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    # # SOLUTION: Activation.
    # fc2    = tf.nn.relu(fc2)
    # dropout
    # hidden_layer = tf.nn.dropout(fc2, keep_prob)
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(400, 43), mean = mu, stddev = sigma),name="fc3_W")
    fc3_b  = tf.Variable(tf.zeros(43),name="fc3_b")
    logits = tf.matmul(fc0, fc3_W) + fc3_b
    
    return logits

# 初始化训练使用的输入输出容器
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None, ))
one_hot_y = tf.one_hot(y, 43)

keep_prob = tf.placeholder(tf.float32)

# 训练模型选择
logits = cnn(x)

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# 其他参数设置
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

print()
print('Settings Done.')
print()

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(train_x)

    print()
    print("Training...")
    print()

    # 每个epoch的训练
    for i in range(EPOCHS):
        x_train, y_train = shuffle(train_x, train_y)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        # 计算validation的accuracy    
        validation_accuracy = evaluate(valid_x, valid_y)
        print("EPOCH {} ...".format(i+1))

        # 准备输出文件
        with open('output.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([validation_accuracy])

        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

        fig_x.append(i)
        fig_y.append(validation_accuracy)
    
    saver.save(sess, './lenet')
    print("Model saved")

# 运行测试集看测试结果
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    sess.run(tf.global_variables_initializer())
    saver1 = tf.train.import_meta_graph('./lenet.meta')
    saver1.restore(sess, "./lenet")
    
    # 计算test的accuracy
    test_accuracy = evaluate(test_x, test_y)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    with open('output.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([test_accuracy])

# 开始画图
h1=ax.plot(fig_x,fig_y,color='lightblue',marker='^',label='line1')
plt.savefig('output.pdf')

sess.close()


        