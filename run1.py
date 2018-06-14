import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, f1_score

fig_w = 45  
batch = 100 

def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # Load the dataset
    X = np.fromfile("mnist_train/mnist_train_data", dtype=np.uint8)
    y = np.fromfile("mnist_train/mnist_train_label", dtype=np.uint8)
    X = X.reshape(-1, fig_w, fig_w, 1)

    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=.3)

    X_test = np.fromfile("mnist_test/mnist_test_data", dtype=np.uint8)
    y_test = np.fromfile("mnist_test/mnist_test_label", dtype=np.uint8)   

    X_test = X_test.reshape(-1, fig_w, fig_w, 1)

    return [X_train, y_train, X_vali, y_vali, X_test, y_test]


def cnn_net(features):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features, [-1, 45, 45, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 45, 45, 1]
    # Output Tensor Shape: [batch_size, 45, 45, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.sigmoid)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 45, 45, 32]
    # Output Tensor Shape: [batch_size, 45, 45, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.sigmoid)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 11, 11, 64]
    # Output Tensor Shape: [batch_size,1 * 11 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 11 * 11 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.sigmoid)

    # Add dropout operation; 0.6 probability that element will be kept
    dense = tf.nn.softmax(dense)
    return dense


class embedModel(object):
    def __init__(self):
        self.X_train, self.y_train, \
            self.X_vali, self.y_vali, \
                self.X_test, self.y_test = load_data()

        self.train_class = []

        for i in range(10):
            arg = np.where(self.y_train == i)[0]
            self.train_class.append(arg)

    def build_network(self):
        self.target = tf.placeholder(shape=[1, fig_w, fig_w, 1], dtype=tf.float32)
        self.sample_sim = tf.placeholder(shape=[None, fig_w, fig_w, 1], dtype=tf.float32)
        self.sample_diff = tf.placeholder(shape=[None, fig_w, fig_w, 1], dtype=tf.float32)

        with tf.variable_scope("cnn") as scope:
            embed_tar = cnn_net(self.target)
        with tf.variable_scope("cnn", reuse=True):
            self.embed_sim = embed_sim = cnn_net(self.sample_sim)
        with tf.variable_scope("cnn", reuse=True):
            embed_diff = cnn_net(self.sample_diff)       

        similarity = tf.matmul(embed_sim, tf.transpose(embed_tar))
        diffference = tf.matmul(embed_diff, tf.transpose(embed_tar))
        
        # make the embeddings of the same labels similar and negative sampling

        self.loss = tf.reduce_sum(diffference) - tf.reduce_sum(similarity)

        self.optimizer = tf.train.AdamOptimizer(0.0000001).minimize(self.loss)   

    def train(self):
        train_num = len(self.X_train)
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True        
        with tf.Session(config=config) as sess:
            sess.run(init_op)        
            total_loss = 0
            for epoch in range(40):
                for i in range(train_num):
                    if i % 10000 == 0:
                        print("epoch: {}, num: {}".format(epoch, i))
                        print("average loss: {}".format(total_loss / 1000))
                        total_loss = 0
                    tar = self.X_train[i]

                    sim_arg = list(self.train_class[self.y_train[i]])
                    diff_arg = []
                    for j in range(10):
                        if j != self.y_train[i]:
                            diff_arg.extend(self.train_class[j])   
                    # get the sampling which are the same and different labels
                    sim = random.sample(sim_arg, batch)
                    diff = random.sample(diff_arg, batch)
                    sim = self.X_train[np.array(sim)]
                    diff = self.X_train[np.array(diff)]

                    feed_dict = {}
                    feed_dict[self.target] = np.expand_dims(tar, 0)
                    feed_dict[self.sample_sim] = sim
                    feed_dict[self.sample_diff] = diff
                    #train on batches
                    op, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    total_loss += loss
                # test
                
                test = self.get_embedding(sess, 'test')
                vali = self.get_embedding(sess, 'vali')
                train = self.get_embedding(sess, 'train')
                
                # print the score and select the best epoch on validation set
                knn = KNN(10)
                knn.fit(train, self.y_train)
                y_vali = knn.predict(vali)
                print("f1 score: {}, accuracy: {}".format(\
                    f1_score(self.y_vali, y_vali, average='micro'), accuracy_score(self.y_vali, y_vali)))

                y_test = knn.predict(test)
                print("f1 score: {}, accuracy: {}".format(\
                    f1_score(self.y_test, y_test, average='micro'), accuracy_score(self.y_test, y_test)))               





            
    def get_embedding(self, sess, mode):
        vec = []
        batch = 1000

        if mode == 'train':
            X = self.X_train
        elif mode == 'vali':
            X = self.X_vali
        else:
            X = self.X_test

        for i in range(0, len(X), batch):
            feed_dict = {}
            feed_dict[self.sample_sim] = X[i:i+batch]

            hidden = sess.run(self.embed_sim, feed_dict=feed_dict)
            vec.append(hidden)
        vec = np.asarray(vec)
        vec = vec.reshape(-1, vec.shape[-1])
        return vec


def main():
    model = embedModel()
    model.build_network()
    model.train()

if __name__ == "__main__":
    main()
