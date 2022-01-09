import tensorflow as tf
import numpy as np

class Net():
    def __init__(self):
        self.learning_rate = 0.01 
        self.x = tf.placeholder(tf.float32, [None, 784], name='InputData')
        # 0-9 digits recognition => 10 classes
        self.y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

        # Set model weights
        self.W = tf.Variable(tf.zeros([784, 10]), name='Weights')
        self.b = tf.Variable(tf.zeros([10]), name='Bias')

        with tf.name_scope('Model'):
             # Model
            self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b) # Softmax
        with tf.name_scope('Loss'):
             # Minimize error using cross entropy
            self.cost = tf.reduce_mean(-tf.reduce_sum( self.y * tf.log(self.pred), reduction_indices=1))
        with tf.name_scope('SGD'):
            # Gradient Descent
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        with tf.name_scope('Accuracy'):
            # Accuracy
            self.acc = tf.equal(tf.argmax(self.pred, 1), tf.argmax( self.y, 1))
            self.acc = tf.reduce_mean(tf.cast(self.acc, tf.float32))

        # # Create a summary to monitor cost tensor
        # tf.summary.scalar("loss", self.cost)
        # # Create a summary to monitor accuracy tensor
        # tf.summary.scalar("accuracy", self.acc)
        # # Merge all summaries into a single op
        # self.merged_summary_op = tf.summary.merge_all()
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", self.acc)
            

if __name__ == "__main__":
    # Parameters
    
    training_epochs = 10
    batch_size = 100
    display_epoch = 1
    tf.set_random_seed(10)
    np.random.seed(10)

    net = Net()
    
    # tf.summary.scalar("loss", net.cost)
    # # Create a summary to monitor accuracy tensor
    # tf.summary.scalar("accuracy", net.acc)


    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    merged_summary_op2 = tf.summary.merge_all()
    
    
    # Initializing the variables
    
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)

        # op to write logs to Tensorboard

        summary_writer = tf.summary.FileWriter('/home/exp/SMGCN/SMGCN/extra/test_tf_summary/', graph=tf.get_default_graph())
        summary_writer2 = tf.summary.FileWriter('/home/exp/SMGCN/SMGCN/extra/test_tf_summary_2/', graph=tf.get_default_graph())
        # x = tf.random_normal(shape=(5,784), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
        # y = tf.random_normal(shape=(5,10), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
        x = np.random.randn(5, 784).astype(np.float32)
        y = np.random.randn(5, 10).astype(np.float32)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = 5
            # Loop over all batches
            for i in range(total_batch):

                # Run optimization op (backprop), cost op (to get loss value)
                # and summary nodes
                _, c, summary = sess.run([net.optimizer, net.cost, merged_summary_op],
                                        feed_dict={net.x: x, net.y: y})
                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * total_batch + i)
                # Compute average loss
                avg_cost += c / total_batch


            for i in range(3):

                # Run optimization op (backprop), cost op (to get loss value)
                # and summary nodes
                _, c, test_summary = sess.run([net.optimizer, net.cost, merged_summary_op],
                                        feed_dict={net.x: x, net.y: y})
                # Write logs at every iteration
                summary_writer2.add_summary(test_summary, epoch * 3 + i)
                # Compute average loss
                avg_cost += c / 3
            # Display logs per epoch step
            if (epoch+1) % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")