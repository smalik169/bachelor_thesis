import tensorflow as tf
import numpy as np


def generate_addition_example(T, batchsize, p=0.3):
	rng = np.random
	x = rng.uniform(size=(batchsize, T)).astype('float32')

	i1 = rng.randint(0,T/2, batchsize)
	i2 = rng.randint(T/2, T, batchsize)
	y = np.zeros((batchsize, T))
	y[np.arange(batchsize), i1] = 1
    	y[np.arange(batchsize), i2] = 1

	X = np.array([x,y], dtype='float32')
	Y = (x*y).sum(axis=1, keepdims=True, dtype='float32')
	
	return X, Y.T


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=1.0/np.sqrt(shape[0]))
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0., shape=shape)
	return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[2, None, None])
y = tf.placeholder(tf.float32, shape=[1, None])

activation = lambda x: tf.maximum(x, .01*x)

### the model
with tf.variable_scope('feedforward_attention'):
    D = 100
    W_xh = weight_variable([D, 2])
    b_xh = bias_variable([D,1])

    h = activation(tf.matmul(W_xh, tf.reshape(x, [2,-1])) + b_xh)

    W_hc = weight_variable([1, D])
    b_hc = bias_variable([1])

    e = tf.reshape( tf.tanh(tf.matmul(W_hc, h) + b_hc), tf.pack( [tf.shape(x)[1], tf.shape(x)[2]] ) )
    #e += tf.random_normal(shape=tf.shape(e), stddev=1.0/tf.to_float(tf.shape(e)[0]), dtype=tf.float32)
    alpha = tf.reshape(tf.exp(e)/tf.reduce_sum(tf.exp(e), 1, keep_dims=True), [1, -1])

    c_tmp = tf.reduce_sum( tf.reshape( alpha * h, tf.pack([-1, tf.shape(x)[2]]) ), 1, keep_dims=True)
    c = tf.reshape(c_tmp, [D, -1])

    W_cs = weight_variable([D, D])
    b_cs = bias_variable([D,1])

    s = activation(tf.matmul(W_cs, c) + b_cs)

    W_sy = weight_variable([1, D])
    b_sy = bias_variable([1])

    y_pred = activation( tf.matmul(W_sy, s) + b_sy )

    mean_square_error = tf.reduce_mean( (y - y_pred)**2 )
###

accuracy = tf.reduce_mean( tf.cast( tf.less( tf.abs(y_pred - y), 0.04), tf.float32 ) )

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_square_error)

#saver = tf.train.Saver()

T = lambda T0: np.random.randint(T0, 1.1*T0)


if __name__ == "__main__":
	session = tf.Session()

	train_set = [generate_addition_example(T(500), 100) for i in xrange(2000)]
	validation_set = [generate_addition_example(T(500), 10) for i in xrange(100)]
	
	session.run(tf.initialize_all_variables())

	####################################################################################################
	saver = tf.train.Saver()
#	saver.restore(session, "model1.ckpt")
	####################################################################################################
	
	prev_train_accuracy = np.array([session.run(accuracy, feed_dict={x:validation_set[j][0], y:validation_set[j][1]}) for j in xrange(100)]).mean()
	prev_err = np.array([session.run(mean_square_error, feed_dict={x:validation_set[j][0], y:validation_set[j][1]}) for j in xrange(100)]).mean()

	for i in range(500000):
    		#T = np.random.randint(T0, 1.1*T0)
		#X, Y = generate_addition_example(T, 100)
    		X, Y = train_set[i%1000]
		
		session.run(train_step, feed_dict={x: X, y: Y})
		
		if i%1000 == 0:
			train_accuracy = np.array([session.run(accuracy, feed_dict={x:validation_set[j][0], y:validation_set[j][1]}) for j in xrange(100)]).mean()
	   		train_error = np.array([session.run(mean_square_error, feed_dict={x:validation_set[j][0], y:validation_set[j][1]}) for j in xrange(100)]).mean()
			#if train_accuracy > prev_train_accuracy:
			if train_error < prev_err:
				#prev_train_accuracy = train_accuracy
				prev_err = train_error
				saver.save(session, "model1.ckpt")
							   
			print("step %d, training accuracy %g, training error %g"%(i, train_accuracy, train_error))
														       

	session.close()
