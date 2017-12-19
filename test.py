import models
import pickle
import numpy as np
import tensorflow as tf






X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])
learning_rate = tf.placeholder("float", [])

# ResNet Models
net = models.resnet(X, 20)
# net = models.resnet(X, 32)
# net = models.resnet(X, 44)
# net = models.resnet(X, 56)

cross_entropy = -tf.reduce_sum(Y*tf.log(net))
opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_op = opt.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


saver = tf.train.Saver()
with tf.Session() as sess:
    checkpoint = tf.train.latest_checkpoint('./res/out')
    saver.restore(sess, checkpoint) 
    acc_totle =sess.run(accuracy,feed_dict={
        X:X_test[:],Y:Y_test[:]})
    print(acc_totle)

