import pickle
import cv2
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import numpy as np

training_file = 'train.p'
testing_file = 'test.p'

train_x = []
train_y = []
test_x = []
test_y = []

print()
print("Reading Data...")
print()

# load data from .p file
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# preprocessing images
for img in train['features']:
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    train_x.append(img)

for img in test['features']:
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    test_x.append(img)

train_y = train['labels']
test_y = test['labels']

# normalization
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

train_x = train_x.reshape(train_x.shape[0], 32, 32, 1)
test_x = test_x.reshape(test_x.shape[0], 32, 32, 1)

train_x = train_x.astype('float32')/255.0   
test_x = test_x.astype('float32')/255.0

# generating validation set from original training set
train_x,valid_x,train_y,valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=random.randint(0,100))

# print out the information of dataset
print("Training datasize: "+str(train_x.shape))
print("Testing datasize: "+str(train_x.shape))
print()

# 学习速率
rate = 0.001
# Epochs 大小
EPOCHS = 25
# batch 大小
BATCH_SIZE = 128


def cnn(x):    
    mu = 0
    sigma = 0.1
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name="conv1_W")
    conv1_b = tf.Variable(tf.zeros(6),name="conv1_b")
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.selu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma),name="conv2_W")
    conv2_b = tf.Variable(tf.zeros(16),name="conv2_b")
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.selu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    fc0   = flatten(conv2)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma), name="fc1_W")
    fc1_b = tf.Variable(tf.zeros(120),name="fc1_b")
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma),name="fc2_W")
    fc2_b  = tf.Variable(tf.zeros(84),name="fc2_b")
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, 0.75)
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma),name="fc3_W")
    fc3_b  = tf.Variable(tf.zeros(43),name="fc3_b")
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    regularizers = (tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc1_b) + tf.nn.l2_loss(fc2_W)+tf.nn.l2_loss(fc2_b)+tf.nn.l2_loss(fc3_W)+tf.nn.l2_loss(fc3_b))
    return logits, regularizers

# 初始化训练使用的输入输出容器
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None, ))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)
factor = 5e-4

# 训练模型选择
logits, regularizers = cnn(x)

# 其他参数设置
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
loss_operation += factor * regularizers
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
softmax=tf.nn.softmax(logits)
prediction = tf.argmax(logits,1)

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]  
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



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

        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

    saver.save(sess, './lenet')
    print("Model saved\n")


# 运行测试集看测试结果
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver1 = tf.train.import_meta_graph('./lenet.meta')
    saver1.restore(sess, "./lenet")
    
    test_accuracy = evaluate(test_x, test_y)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
