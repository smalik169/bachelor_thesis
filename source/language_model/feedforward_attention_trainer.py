import numpy as np
import tensorflow as tf
from ff_language_models import *

class Trainer(object):
	def __init__(self, session, model, weight_decay=10**(-5), initial_lr=0.1, lr_decay=10**(-8)):
		self.sess = session
		self.model = model
		self.initial_lr = initial_lr
		self.lr_decay = lr_decay
		self.time_step = 0

		self.nll = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(model.logits, model.y) )
		self.loss = self.nll + weight_decay*( tf.reduce_sum( model.W_ih**2 ) + tf.reduce_sum( model.W_ho**2 ) )/2.0

		self.learning_rate = tf.placeholder(tf.float32)
		self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.85).minimize(self.loss)
		
		self.sess.run(tf.initialize_all_variables())
	
	def make_step(self, mini_batch, lr=None):
		if lr is None:
			lr = self.initial_lr/(1 + self.lr_decay*self.time_step)
		

		mb_nll, mb_loss = self.sess.run((self.nll, self.loss, self.train_step), feed_dict={self.model.x:mini_batch[:, :-1], self.model.y:mini_batch[:, -1], self.learning_rate:lr})[:2]
		self.time_step += 1
		return mb_nll, mb_loss
		

def execute_on(sess, model, node, data_set, batch_size):
	return np.array([sess.run(node, feed_dict={model.x:de[:, :-1], model.y:de[:, -1]}) for j in xrange(0, data_set.shape[0], batch_size) for de in [data_set[j:j+batch_size]]])

mean = lambda arr: 1.0*arr.sum()/arr.ravel().shape[0]


if __name__ == "__main__":
	file_name = raw_input("filename to save results: ")

	n_gram = 5 
	prefix_size = 5

	word2number = dict()
	number2word = dict()

	with open("/pio/scratch/1/i264266/licencjat/language_model/wiki_stream_data/words.txt", 'rb') as f:
		number = 0
		for line in f:
			word = line.strip()
			
			word2number[word] = number
			number2word[number] = word
			
			number += 1

	data = np.load("/pio/scratch/1/i264266/licencjat/language_model/wiki_stream_data/word_stream.npy")

	data_entry_size = n_gram + prefix_size
	train_data = np.array([data[i:i+data_entry_size] for i in range(5000000-data_entry_size)])
	validation_data = np.array([data[i:i+data_entry_size] for i in range(5000000, 5500000-data_entry_size)])
	test_data = np.array([data[i:i+data_entry_size] for i in range(5500000,6000000-data_entry_size)])

	session = tf.Session()
	
	net = AttentionFFLM(vocab_size=len(word2number), embed_size=100, hid_units=200, n_gram=n_gram)
	trainer = Trainer(session, net, initial_lr=0.15, lr_decay=10**(-5))

	saver = tf.train.Saver()

	i = 0
	epoch_number = 0
	batch_size = 1000

	train_data = train_data[:batch_size * (train_data.shape[0]/batch_size)]
	validation_data = validation_data[:batch_size * (validation_data.shape[0]/batch_size)]
	test_data = test_data[:batch_size * (test_data.shape[0]/batch_size)]

	train_log = []
	validation_log = []
	prev_best_loss = np.inf
	prev_best_nll = np.inf

	####
#	saver.restore(session, file_name+".ckpt")
#	train_log, validation_log = np.load( file_name + ".npy" )[:2]
#	old_val = np.array(validation_log).T[0]
#	trainer.time_step= (train_data.shape[0]/batch_size)*old_val.shape[0]
#	prev_best_nll = old_val[-1]
#	epoch_number = old_val.shape[0]
	####

	while True:
		mini_batch = train_data[i:i+batch_size]
		mini_batch_nll, mini_batch_loss = trainer.make_step(mini_batch)
	    
	   	train_log.append( [mini_batch_nll, mini_batch_loss] )
	    
	   	if i%(1000*batch_size) == 0:
			print "epoch:", epoch_number, "step:", i/batch_size, "loss:", mini_batch_loss, "nll:", mini_batch_nll
		
		i = (i + batch_size)%train_data.shape[0]
		
#		if i < batch_size: i = 0;

		if i == 0:
			validation_loss = mean( execute_on(session, net, trainer.loss, validation_data, batch_size) )
			validation_nll = mean( execute_on(session, net, trainer.nll, validation_data, batch_size) )
			validation_log.append( [validation_nll, validation_loss] )
			
			if prev_best_nll > validation_nll:
				saver.save(session, file_name+".ckpt")
				prev_best_nll = validation_nll
			
			epoch_number += 1
			print "after epoch:", epoch_number-1, "mean_validation_loss:", validation_loss, "mean_validation_nll:", validation_nll
		
		if epoch_number == 25:
		    break

	print "restoring best model..."
	saver.restore(session, file_name+".ckpt")
				
	test_loss = mean( execute_on(session, net, trainer.loss, test_data, batch_size) )
	test_nll = mean( execute_on(session, net, trainer.nll, test_data, batch_size) )

	print "after", epoch_number, "epochs: test_loss:", test_loss, "test_nll:", test_nll

	print "saving results..."
	np.save( file_name, np.array( [train_log, validation_log, [test_nll, test_loss]] ) )
