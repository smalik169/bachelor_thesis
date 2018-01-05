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

print "train_set.shape =", train_set.shape
print "test_set.shape =", test_set.shape

activation = lambda x: tf.maximum(x, .01*x)

with tf.variable_scope('feedforward_attention'):
### the model
    x = tf.placeholder(tf.float32, shape=[2, None, None])
    y = tf.placeholder(tf.float32, shape=[1, None])

    D = 100
    W_xh = weight_variable([D, 2])
    b_xh = bias_variable([D,1])

    h = activation(tf.matmul(W_xh, tf.reshape(x, [2,-1])) + b_xh)

    W_hc = weight_variable([1, D])
    b_hc = bias_variable([1])

    e = tf.reshape( tf.tanh(tf.matmul(W_hc, h) + b_hc), tf.pack( [tf.shape(x)[1], tf.shape(x)[2]] ) )
    #e += tf.random_normal(shape=tf.shape(e), stddev=1.0/tf.to_float(tf.shape(e)[0]), dtype=tf.float32)
    alpha = tf.reshape(tf.exp(e)/tf.reduce_sum(tf.exp(e), 1, keep_dims=True), [1, -1])

#    c_tmp = tf.reduce_sum( tf.reshape( alpha * h, tf.pack([-1, tf.shape(x)[2]]) ), 1, keep_dims=True)
#    c = tf.reshape(c_tmp, [D, -1])

    W_cs = weight_variable([D, D])
    b_cs = bias_variable([D,1])

#    s = activation(tf.matmul(W_cs, c) + b_cs)

    W_sy = weight_variable([1, D])
    b_sy = bias_variable([1])

#    y_pred = activation( tf.matmul(W_sy, s) + b_sy )

#    mean_square_error = tf.reduce_mean( (y - y_pred)**2 )

#    accuracy = tf.reduce_mean( tf.cast( tf.less( tf.abs(y_pred - y), 0.04), tf.float32 ) )


with tf.variable_scope('lstm_attention'):
    x_rnn = tf.placeholder(tf.float32, shape=[None, None, 2])
    y_rnn = tf.placeholder(tf.float32, shape=[1, None])
    weights_rnn = tf.placeholder(tf.float32, shape=[None, None, 1])

    cell_size = 64
    lstm_cell = rnn_cell.BasicLSTMCell(cell_size, state_is_tuple=True)

    #initial_state = lstm_cell.zero_state(tf.shape(x)[1], tf.float32)
    _, states_rnn = rnn.dynamic_rnn(lstm_cell, x_rnn, dtype=tf.float32, time_major=True, input_weights=weights_rnn)

    final_states_rnn = states_rnn[1]

    W_rnn = weight_variable([1, cell_size])
    b_rnn = bias_variable([1])

    y_pred_rnn = activation( tf.matmul(W_rnn, tf.transpose(final_states_rnn)) + b_rnn )

    mean_square_error_rnn = tf.reduce_mean( (y_rnn - y_pred_rnn)**2 )

    accuracy_rnn = tf.reduce_mean( tf.cast( tf.less( tf.abs(y_pred_rnn - y_rnn), 0.04), tf.float32 ) )

    #train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_square_error)
    
    optimizer_rnn = tf.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars_rnn = optimizer_rnn.compute_gradients(mean_square_error_rnn)
    clipped_grads_and_vars_rnn = [(tf.clip_by_norm(grad, 1.), var) for (grad, var) in grads_and_vars_rnn \
                                  if not var.name.startswith('feedforward_attention')]
    train_step_rnn = optimizer_rnn.apply_gradients(clipped_grads_and_vars_rnn)

if __name__ == "__main__":
    session = tf.Session()
    saver = tf.train.Saver([v for v in tf.all_variables() if v.name.startswith('feedforward_attention')])
    
    for i in xrange(5):
        session.run(tf.initialize_all_variables())
        
        saver.restore(session, "model1.ckpt")
    
        train_log = []
        for j in xrange(15000):
            X, Y = train_set[j%2000]
    
            weights = session.run(e, feed_dict={x: X.T, y: Y})
            weights = weights.T.reshape( X.shape[:2] +(1,) )
            train_log.append((session.run(accuracy_rnn, feed_dict={x_rnn: X, y_rnn: Y, weights_rnn:weights}), \
                              session.run(mean_square_error_rnn, feed_dict={x_rnn: X, y_rnn: Y, weights_rnn:weights})))
    
            session.run(train_step_rnn, feed_dict={x_rnn: X, y_rnn: Y, weights_rnn:weights})
        
            if j%100 == 0:
                print 'run', i, 'step', j, train_log[-1]
    
        print 'testing...'
        err_list = []
        for (Xt, Yt) in test_set:
            weights = session.run(e, feed_dict={x: Xt.T, y: Yt})
            weights = weights.T.reshape( Xt.shape[:2] +(1,) )
            err_list.append( session.run(mean_square_error_rnn, feed_dict={x_rnn: Xt, y_rnn: Yt, weights_rnn:weights}) )

    
        print 'saving results...'
        np.save('logs/lstm_attention_train_log' + str(i) + '.npy', np.array(train_log))
        np.save('logs/lstm_attention_test_err' + str(i) + '.npy', np.array(err_list))
    
        print
    
    session.close()
