from data import create_dataset
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer


train_x,train_y,ret_train_Y,test_x,test_y,ret_test_Y=create_dataset()
#train_x, train_y, test_x, test_y = create_dataset()
#print(train_x.head(),test_x.head())
n_nodes_hl1 = 80
n_nodes_hl2 = 150
n_nodes_hl3 = 80

n_classes = 40
batch_size = 10000
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum': n_nodes_hl3,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# Nothing changes
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels= y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        saver.save(sess,'./model.ckpt')
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
#
#
# def use_neural_network(input_data):
#     prediction = neural_network_model(x)
#     with open('lexicon.pickle', 'rb') as f:
#         lexicon = pickle.load(f)
#     saver=tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver.restore(sess,"./model.ckpt")
#         current_words = word_tokenize(input_data.lower())
#         lemmatizer = WordNetLemmatizer()
#         current_words = [lemmatizer.lemmatize(i) for i in current_words]
#         features = np.zeros(len(lexicon))
#
#         for word in current_words:
#             if word.lower() in lexicon:
#                 index_value = lexicon.index(word.lower())
#                 # OR DO +=1, test both
#                 features[index_value] += 1
#
#         features = np.array(list(features))
#         # pos: [1,0] , argmax: 0
#         # neg: [0,1] , argmax: 1
#         result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1)))
#         print(result)
#         if result[0] == 1:
#             print('Spam:', input_data)
#             return 1
#         elif result[0] == 2:
#             print('Ham:', input_data)
#             return 2
#         else:
#             print('Info:',input_data)
#             return 3

train_neural_network(x)
