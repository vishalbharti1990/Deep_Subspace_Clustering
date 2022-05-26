
from __future__ import division, print_function, absolute_import
from tabnanny import verbose
import statistics
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
from pathlib import Path
from sklearn import cluster
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from munkres import Munkres
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from time import time
# from tensorflow.examples.tutorials.mnist import input_data


class ConvAE(object):
	def __init__(self, n_input, kernel_size, n_hidden, reg_const1 = 1.0, reg_const2 = 1.0, reg = None, batch_size = 256,\
		denoise = False, model_path = None, logs_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/COIL20CodeModel/new/pretrain/logs'):	
	#n_hidden is a arrary contains the number of neurals on every layer
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.reg = reg
		self.model_path = model_path		
		self.kernel_size = kernel_size		
		self.iter = 0
		self.batch_size = batch_size
		weights = self._initialize_weights()
		
		# model
		self.x = tf.compat.v1.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
		self.learning_rate = tf.compat.v1.placeholder(tf.float32, [])
		
		if denoise == False:
			x_input = self.x
			latent, shape = self.encoder(x_input, weights)

		else:
			x_input = tf.add(self.x, tf.random.normal(shape=tf.shape(input=self.x),
											   mean = 0,
											   stddev = 0.2,
											   dtype=tf.float32))

			latent,shape = self.encoder(x_input, weights)
		self.z_conv = tf.reshape(latent,[batch_size, -1])		
		self.z_ssc, Coef = self.selfexpressive_moduel(batch_size)	
		self.Coef = Coef						
		latent_de_ft = tf.reshape(self.z_ssc, tf.shape(input=latent))		
		self.x_r_ft = self.decoder(latent_de_ft, weights, shape)		
				

		self.saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.trainable_variables() if not (v.name.startswith("Coef"))]) 
			
		
		self.cost_ssc = 0.5*tf.reduce_sum(input_tensor=tf.pow(tf.subtract(self.z_conv,self.z_ssc), 2))
		self.recon_ssc =  tf.reduce_sum(input_tensor=tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
		self.reg_ssc = tf.reduce_sum(input_tensor=tf.pow(self.Coef,2))
		tf.compat.v1.summary.scalar("ssc_loss", self.cost_ssc)
		tf.compat.v1.summary.scalar("reg_lose", self.reg_ssc)		
		
		self.loss_ssc = self.cost_ssc*reg_const2 + reg_const1*self.reg_ssc + self.recon_ssc

		self.merged_summary_op = tf.compat.v1.summary.merge_all()		
		self.optimizer_ssc = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_ssc)
		self.init = tf.compat.v1.global_variables_initializer()
		self.sess = tf.compat.v1.InteractiveSession()
		self.sess.run(self.init)
		self.summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())

	def _initialize_weights(self):
		all_weights = dict()
		all_weights['enc_w0'] = tf.compat.v1.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, n_hidden[0]],
			initializer=tf.keras.initializers.GlorotUniform(),regularizer = self.reg)
		all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))
		

		all_weights['dec_w0'] = tf.compat.v1.get_variable("dec_w0", shape=[self.kernel_size[0], self.kernel_size[0],1, n_hidden[0]],
			initializer=tf.keras.initializers.GlorotUniform(),regularizer = self.reg)
		all_weights['dec_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))
		return all_weights


	# Building the encoder
	def encoder(self,x, weights):
		shapes = []
		# Encoder Hidden layer with relu activation #1
		shapes.append(x.get_shape().as_list())
		layer1 = tf.nn.bias_add(tf.nn.conv2d(input=x, filters=weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
		layer1 = tf.nn.relu(layer1)
		return  layer1, shapes

	# Building the decoder
	def decoder(self,z, weights, shapes):
		# Encoder Hidden layer with relu activation #1
		shape_de1 = shapes[0]
		layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(input=self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
		 strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
		layer1 = tf.nn.relu(layer1)
		
		return layer1



	def selfexpressive_moduel(self,batch_size):
		
		Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')			
		z_ssc = tf.matmul(Coef,	self.z_conv)
		return z_ssc, Coef


	def finetune_fit(self, X, lr):
		C,l1_cost, l2_cost, summary, _ = self.sess.run((self.Coef, self.reg_ssc, self.cost_ssc, self.merged_summary_op, self.optimizer_ssc), \
													feed_dict = {self.x: X, self.learning_rate: lr})
		self.summary_writer.add_summary(summary, self.iter)
		self.iter = self.iter + 1
		return C, l1_cost,l2_cost 
	
	def initlization(self):
		tf.compat.v1.reset_default_graph()
		self.sess.run(self.init)	

	def transform(self, X):
		return self.sess.run(self.z_conv, feed_dict = {self.x:X})

	def save_model(self):
		save_path = self.saver.save(self.sess,self.model_path)
		print ("model saved in file: %s" % save_path)

	def restore(self):
		self.saver.restore(self.sess, self.model_path)
		print ("model restored")
		
class newConvAE20(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_const1 = 1.0, reg_const2 = 1.0, reg = None, batch_size = 256,\
        denoise = False, model_path = None, logs_path = './COIL20CodeModel/new/pretrain/logs',no=0):  
    #n_hidden is a arrary contains the number of neurals on every layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path        
        self.kernel_size = kernel_size      
        self.iter = 0
        self.batch_size = batch_size
        weights = self._initialize_weights()
        self.no = 0 
        # model
        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, [])
        
        if denoise == False:
            x_input = self.x
            latent, pool1, shape = self.encoder(x_input, weights)

        else:
            x_input = tf.add(self.x, tf.random.normal(shape=tf.shape(input=self.x),
                                               mean = 0,
                                               stddev = 0.2,
                                               dtype=tf.float32))

            latent,shape = self.encoder(x_input, weights)

        latent = tf.add(latent,pool1)

        self.z_conv = tf.reshape(latent,[batch_size, -1])       
        self.z_ssc, Coef = self.selfexpressive_moduel(batch_size)   
        self.Coef = Coef                        
        latent_de_ft = tf.reshape(self.z_ssc, tf.shape(input=latent))     
        self.x_r_ft = self.decoder(latent_de_ft, weights, shape)        
                

        self.saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.trainable_variables() if not (v.name.startswith("Coef"))],max_to_keep =None) 
            
        
        self.cost_ssc = 0.5*tf.reduce_sum(input_tensor=tf.pow(tf.subtract(self.z_conv,self.z_ssc), 2))
        self.recon_ssc =  tf.reduce_sum(input_tensor=tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
        self.reg_ssc = tf.reduce_sum(input_tensor=tf.pow(self.Coef,2))
        tf.compat.v1.summary.scalar("ssc_loss", self.cost_ssc)
        tf.compat.v1.summary.scalar("reg_lose", self.reg_ssc)     
        
        self.loss_ssc = self.cost_ssc*reg_const2 + reg_const1*self.reg_ssc + self.recon_ssc

        self.merged_summary_op = tf.compat.v1.summary.merge_all()     
        self.optimizer_ssc = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_ssc)
        self.init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(self.init)
        self.summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['enc_w0'] = tf.compat.v1.get_variable("enc_w0", shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
            initializer=tf.keras.initializers.GlorotUniform(),regularizer = self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype = tf.float32))

        
        all_weights['oenc_w0'] = tf.compat.v1.get_variable("oenc_w0", shape=[self.kernel_size[0], self.kernel_size[0],  self.n_hidden[0],1],
            initializer=tf.keras.initializers.GlorotUniform(),regularizer = self.reg)
        all_weights['oenc_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))

        all_weights['dec_w0'] = tf.compat.v1.get_variable("dec_w0", shape=[self.kernel_size[0], self.kernel_size[0],1, self.n_hidden[0]],
            initializer=tf.keras.initializers.GlorotUniform(),regularizer = self.reg)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([1], dtype = tf.float32))


        return all_weights


    # Building the encoder
    def encoder(self,x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        shapes_en = shapes[0]

        layer1 = tf.nn.bias_add(tf.nn.conv2d(input=x, filters=weights['enc_w0'], strides=[1,2,2,1],padding='SAME'),weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        # print("here")
        # print(shapes_en)
        olayer1 = tf.add(tf.nn.conv2d_transpose(x, weights['oenc_w0'],tf.stack([tf.shape(input=self.x)[0],shapes_en[1]*2,64,self.n_hidden[0]]),\
            strides=[1,2,2,1],padding='SAME'),weights['oenc_b0'])
        olayer1 = tf.nn.relu(olayer1)
        pool1 = tf.compat.v1.layers.max_pooling2d(inputs=olayer1, pool_size=[2,2], strides=4)

        print(layer1.shape,pool1.shape)
        return  layer1, pool1,shapes

    # Building the decoder
    def decoder(self,z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[0]
        # shape_de1 = shapes[2]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack([tf.shape(input=self.x)[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
         strides=[1,2,2,1],padding='SAME'),weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)

        # print(layer1.shape)
        

        return layer1

    def reconstruct(self,X):
        print(self.x_r_ft.shape)
        print(self.x.shape)
        return self.sess.run(self.x_r_ft, feed_dict = {self.x:X})

    def selfexpressive_moduel(self,batch_size):
        
        Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size],tf.float32), name = 'Coef')          
        z_ssc = tf.matmul(Coef, self.z_conv)
        return z_ssc, Coef


    def finetune_fit(self, X, lr):
        C,l1_cost, l2_cost, summary, _ = self.sess.run((self.Coef, self.reg_ssc, self.cost_ssc, self.merged_summary_op, self.optimizer_ssc), \
                                                    feed_dict = {self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return C, l1_cost,l2_cost 
    
    def initlization(self):
        tf.compat.v1.reset_default_graph()
        self.sess.run(self.init)    

    def transform(self, X):
        return self.sess.run(self.z_conv, feed_dict = {self.x:X})

    def save_model(self):
        self.no = self.no+1
        savetmp = self.model_path + "%d.ckpt"%(self.no)
        # save_path = self.saver.save(self.sess,self.model_path)
        # tf.train.Saver(max_to_keep=None)
        # self.saver(max_to_keep = None)
        # self.saver(max_to_keep=None)
        save_path = self.saver.save(self.sess, savetmp )
        # print ("model saved in file: %s" % save_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.model_path)
        print ("model restored")

def best_map(L1,L2):
	#L1 should be the labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass,nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]
	return newL2

def thrC(C,ro):
	if ro < 1:
		N = C.shape[1]
		Cp = np.zeros((N,N))
		S = np.abs(np.sort(-np.abs(C),axis=0))
		Ind = np.argsort(-np.abs(C),axis=0)
		for i in range(N):
			cL1 = np.sum(S[:,i]).astype(float)
			stop = False
			csum = 0
			t = 0
			while(stop == False):
				csum = csum + S[t,i]
				if csum > ro*cL1:
					stop = True
					Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
				t = t + 1
	else:
		Cp = C

	return Cp

def post_proC(C, K, d, alpha):
	# C: coefficient matrix, K: number of clusters, d: dimension of each subspace
	n = C.shape[0]
	C = 0.5*(C + C.T)	 
	C = C - np.diag(np.diag(C)) + np.eye(n,n) # for sparse C, this step will make the algorithm more numerically stable
	r = d*K + 1		
	U, S, _ = svds(C,r,v0 = np.ones(n), tol=1e-07)
	U = U[:,::-1] 
	S = np.sqrt(S[::-1])
	S = np.diag(S)
	U = U.dot(S)
	U = normalize(U, norm='l2', axis = 1)  
	Z = U.dot(U.T)
	Z = Z * (Z>0)
	L = np.abs(Z ** alpha)
	L = L/L.max()
	L = 0.5 * (L + L.T)	
	print("Running Spectral Clustering...")
	spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize', verbose=True, n_jobs=-1)
	spectral.fit(L)
	grp = spectral.fit_predict(L) + 1
	return grp, L

def err_rate(gt_s, s):
	c_x = best_map(gt_s,s)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	return missrate  

def run_odsc(epochs):
	curr_dir = str(Path(__file__).parent)
	data = sio.loadmat(f'{curr_dir}/.././Data/COIL20.mat')
	Img = data['fea']
	Label = data['gnd']
	Img = np.reshape(Img,(Img.shape[0],32,32,1))

	n_input = [32,32]
	kernel_size = [3]
	n_hidden = [15]
	batch_size = 20*72
	# model_path = './pretrain-model-COIL20/model.ckpt'
	# ft_path = '/home/pan/workspace-eclipse/deep-subspace-clustering/COIL20CodeModel/new/pretrain/model.ckpt'
	logs_path = f'{curr_dir}/ft/logs'

	num_class = 20 #how many class we sample
	num_sa = 72

	batch_size_test = num_sa * num_class


	iter_ft = 0
	ft_times = epochs
	display_step = 1
	alpha = 0.04
	learning_rate = 1e-3

	reg1 = 1.0
	reg2 = 15.0
	best_m = 0
	bestep= 0
	# for i in range(1,50):
		
	model_path = f'{curr_dir}/pretrained/modelovern' + "1.ckpt"

	CAE = newConvAE20(n_input = n_input, n_hidden = n_hidden, reg_const1 = reg1, reg_const2 = reg2, kernel_size = kernel_size, \
				batch_size = batch_size_test, model_path = model_path, logs_path= logs_path)

	acc_= []
	ami_, nmi_ = [], []
	stime = time()
	for j in range(0,10):
		print(f"Running repetition {j+1}/10")
		coil20_all_subjs = Img
		coil20_all_subjs = coil20_all_subjs.astype(float)	
		label_all_subjs = Label
		label_all_subjs = label_all_subjs - label_all_subjs.min() + 1    
		label_all_subjs = np.squeeze(label_all_subjs) 
			
		CAE.initlization()
		CAE.restore()
		
		for iter_ft  in range(ft_times):
			iter_ft = iter_ft+1
			C,l1_cost,l2_cost = CAE.finetune_fit(coil20_all_subjs,learning_rate)
			if iter_ft % display_step == 0:
				print("epoch: %.1d" % iter_ft, "cost: %.8f" % (l1_cost/float(batch_size_test)))
				C = thrC(C,alpha)
				y_x, CKSym_x = post_proC(C, num_class, 12 ,8)
				missrate_x = err_rate(label_all_subjs,y_x)
				acc = 1 - missrate_x
				print ("experiment: %d" % iter_ft)
				m=acc
		acc_.append(acc)
		ami_.append(adjusted_mutual_info_score(label_all_subjs, y_x))
		# nmi = normalized_mutual_info_score(label_all_subjs, y_x)
	print(f"Time for 10 reps = {time()-stime}")
	return acc_, ami_ 

if __name__=="__main__":
	train_epochs = 40
	acc, ami = run_odsc(train_epochs)
	print(f"Accuracy = {round(statistics.mean(acc), 4)} \u00B1 {round(statistics.stdev(acc), 4)}")
	print(f"AMI = {round(statistics.mean(ami), 4)} \u00B1 {round(statistics.stdev(ami), 4)}")