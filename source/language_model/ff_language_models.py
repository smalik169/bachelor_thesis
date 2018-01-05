import numpy as np
import tensorflow as tf


def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=1.0/np.sqrt(sum(shape)))
        return tf.Variable(initial)

def bias_variable(shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)


class FFLM(object):
	def __init__(self, vocab_size, embed_size, hid_units, n_gram, activation=tf.nn.relu):
		self.x = tf.placeholder(tf.int32, shape=[None, n_gram-1])
		self.y = tf.placeholder(tf.int32, shape=[None])
		
		self.embedding = weight_variable(shape=(vocab_size, embed_size))

		self.get_them = tf.nn.embedding_lookup(self.embedding, self.x)
		self.encoded_input = tf.reshape( self.get_them, (-1, (n_gram-1)*embed_size) )

		self.W_ih = weight_variable(((n_gram-1)*embed_size, hid_units))
		self.b_ih = bias_variable((1, hid_units))

		self.hid = activation( tf.matmul(self.encoded_input, self.W_ih) + self.b_ih )

		self.W_ho = weight_variable((hid_units, vocab_size))
		self.b_ho = bias_variable((1, vocab_size))

		self.logits = tf.matmul(self.hid, self.W_ho) + self.b_ho

		self.output_probs = tf.nn.softmax(self.logits)


class AttentionFFLM(object):
	def __init__(self, vocab_size, embed_size, hid_units, n_gram, activation=tf.nn.relu, distance=False):
		self.x = tf.placeholder(tf.int32, shape=[None, None])
		self.y = tf.placeholder(tf.int32, shape=[None])

		self.embedding = weight_variable(shape=(vocab_size, embed_size))

		self.get_them = tf.nn.embedding_lookup(self.embedding, self.x)
		self.cut = tf.shape(self.x)[1]-(n_gram-1)

		self.slice_ngram_start = tf.pack([0, self.cut, 0])
		self.slice_ngram_size = tf.pack([tf.shape(self.get_them)[0], n_gram-1, tf.shape(self.get_them)[2]])
		self.encoded_ngram = tf.reshape( tf.slice( self.get_them, self.slice_ngram_start, self.slice_ngram_size ), [-1, (n_gram-1)*embed_size] )

		self.slice_prefix_start = [0,0,0]
		self.slice_prefix_size = tf.pack([tf.shape(self.get_them)[0], self.cut, tf.shape(self.get_them)[2]])
		self.prefix = tf.slice( self.get_them, self.slice_prefix_start, self.slice_prefix_size )
		
		if distance:
			##distance stuff
			self.num = tf.reshape(tf.linspace(tf.to_float(self.cut), 1.0, self.cut), [-1,1])

			self.W_nd = weight_variable([1, 10])
			self.b_nd = bias_variable([1, 10])

			self.dist = activation( tf.matmul(self.num, self.W_nd) + self.b_nd )

			self.W_dh = weight_variable([10, embed_size])
			self.b_dh = bias_variable([1, embed_size])

			self.mod_h = tf.reshape(activation( tf.matmul(self.dist, self.W_dh) + self.b_dh ), [1, -1])
		else:
			
			self.mod_h = tf.constant(0.0)

		##attention stuff
		self.W_ph = weight_variable([embed_size, embed_size])
		self.b_ph = bias_variable([1, embed_size])

		self.h = activation( tf.matmul(tf.reshape(self.prefix, [-1, embed_size]), self.W_ph) + self.b_ph )
		self.h_ = tf.reshape( tf.reshape(self.h, tf.pack([-1, self.cut*embed_size])) + self.mod_h, tf.shape(self.h))

		self.W_ha = weight_variable([embed_size, 1])
		self.b_ha =bias_variable([1])

		self.att_logits = tf.reshape( tf.tanh(tf.matmul(self.h_, self.W_ha) + self.b_ha), tf.pack([-1, self.cut]) )
		self.alpha = tf.reshape(tf.nn.softmax(self.att_logits), [1,-1])


		self.att_tmp = tf.reduce_sum( tf.reshape( self.alpha * tf.transpose(self.h), tf.pack([-1, self.cut] )), 1, keep_dims=True)
		self.att = tf.transpose( tf.reshape(self.att_tmp, [embed_size, -1]) )

		self.input_w_attention = tf.concat(1, [self.att, self.encoded_ngram])
		##
		
		self.W_ih = weight_variable(((1 + n_gram-1)*embed_size, hid_units))
		self.b_ih = bias_variable((1, hid_units))

		self.hid = activation( tf.matmul(self.input_w_attention, self.W_ih) + self.b_ih )

		self.W_ho = weight_variable((hid_units, vocab_size))
		self.b_ho = bias_variable((1, vocab_size))

		self.logits = tf.matmul(self.hid, self.W_ho) + self.b_ho
		self.output_probs = tf.nn.softmax(self.logits)

