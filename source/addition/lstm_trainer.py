import tensorflow as tf
import numpy as np
import rnn_cell, rnn

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0/np.sqrt(shape[0]))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

train_set = np.load('train_set.npy')
test_set = np.load('test_set.npy')

x = tf.placeholder(tf.float32, shape=[None, None, 2])
y = tf.placeholder(tf.float32, shape=[1, None])

activation = lambda x: tf.maximum(x, .01*x)

with tf.variable_scope('lstm'):
    cell_size = 64
    lstm_cell = rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=True)

    _, states = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, time_major=True)

    final_states = states[1]

    W = weight_variable([1, cell_size])
    b = bias_variable([1])

    y_pred = activation( tf.matmul(W, tf.transpose(final_states)) + b )

    mean_square_error = tf.reduce_mean( (y - y_pred)**2 )

    accuracy = tf.reduce_mean( tf.cast( tf.less( tf.abs(y_pred - y), 0.04), tf.float32 ) )

    #train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_square_error)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars = optimizer.compute_gradients(mean_square_error)
    clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1.), var) for (grad, var) in grads_and_vars]
    train_step = optimizer.apply_gradients(clipped_grads_and_vars)


if __name__ == "__main__":
    session = tf.Session()

    for i in xrange(5):
        session.run(tf.initialize_all_variables())
    
        train_log = []
        for j in xrange(15000):
            X, Y = train_set[j%2000]
    
            train_log.append((session.run(accuracy, feed_dict={x: X, y: Y}), \
                              session.run(mean_square_error, feed_dict={x: X, y: Y})))
    
            session.run(train_step, feed_dict={x: X, y: Y})
    
            if j%100 == 0:
                print 'run', i, 'step', j, train_log[-1]
    
        print 'testing...'
        err_list = []
        for (Xt, Yt) in test_set:
            err_list.append( session.run(mean_square_error, feed_dict={x: Xt, y: Yt}) )
    
        print 'saving results...'
        np.save('logs/lstm_train_log' + str(i) + '.npy', np.array(train_log))
        np.save('logs/lstm_test_err' + str(i) + '.npy', np.array(err_list))
    
        print
    
    session.close()
