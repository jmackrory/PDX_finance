"""
Recurent Neural Network

Creates python object which stores tensorflow graph.
Makes a simple multilayer RNN, with mapping from inputs 
to hidden layers.  Currently outputs just the final set of prices.

Based on object oriented framework used in CS 224,
and A. Geron's "Hands on Machine Learning with Scikit-Learn and
Tensorflow" Ch 14 on RNN.  
Check those out, I stole pretty liberally from Geron.
The tensorflow docs are pretty rough, but the tutorials are almost
readable. 

The input (X), and target (y) placeholders are defined in add_placeholders.
These are inputs to the TF graph.

Before the network can be run it should be built, which defines the graph.

The guts of the network are defined in add_prediction_op, which
has an input/output hidden layer to reduce dimension.  
There is then a multilayer, dynamic RNN inside.
This is all defined with tensorflow intrinsics/added modules.
Currently, I've turned off the dropout, which should only be active
during training.  

The training is done with batch gradient descent optimization
via the Adam optimizer which is an improved Gradient descent.
It scales gradients, includes a momentum variable).
Note that tensorflow handles backpropagation automatically. 

The loss/cost function is defined in add_loss_op, and is just 
the mean-square error across stocks.

Prediction and inference is done in predict_all()
In order to do prediction/inference, a model is loaded from a saved file
(with graph defined in a Metagraph, and variables loaded via Saver).

Currently data is read in via feed_dict, which is super slow.
Apparently tf.Data is the new preferred simple framework for this.
"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn import MultiRNNCell, BasicRNNCell, GRUCell, LSTMCell,\
    DropoutWrapper

import numpy as np
import matplotlib.pyplot as plt

#to prevent creating huge logs.
from IPython.display import clear_output
import time

class recurrent_NN(object):
    """
    Make a multi-layer recurrent neural network for predicting next days
    stock data.

    """
    def __init__(self,Nsteps,Ninputs,Nhidden,Noutputs,cell):
        """
        Initialize model and build initial graph.

        Nsteps - number of time steps
        Ninputs - number of input stocks/features
        Nhidden - number of hidden degrees of freedom
        Noutputs - number of outputs at final time-step
        cell - type of recurrent cell to use (basic, LSTM, GRU)
        """
        #number of outputs per input
        self.Noutputs=Noutputs
        self.Ninputs=Ninputs        
        #number of steps
        self.Nsteps=Nsteps
        #number of dim on input
        self.cell_type=cell
        self.Nlayers=2
        self.Nhidden=Nhidden
        self.lr = 0.001
        self.keep_prob=0.5
        self.n_iter=200
        self.nprint=20
        self.is_training=True
        self.is_dropout=False
        #only grabbing a fraction of the data
        self.Nbatch=100
        #makes the tensor flow graph.
        self.build()

    def build(self):
        """Creates essential components for graph, and 
        adds variables to instance. 
        """
        tf.get_default_graph()
        tf.reset_default_graph()        
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        """Adds input, output placeholders to graph. 
        Note that these are python object attributes.
        """
        #load in the training examples, and their labels
        #inputs:  Nobs, with n_steps, and n_inputs per step
        self.X = tf.placeholder(tf.float32,[None,self.Nsteps,self.Ninputs],name='X')
        #Outputs: n_outputs we want to predict in the future.
        self.y = tf.placeholder(tf.float32,[None,self.Noutputs],name='y')

    def create_feed_dict(self,inputs_batch, labels_batch=None):
        """Make a feed_dict from inputs, labels as inputs for 
        graph.
        Args:
        inputs_batch - batch of input data
        label_batch  - batch of output labels. (Can be none for prediction)
        Return:
        Feed_dict - the mapping from data to placeholders.
        """
        feed_dict={self.X:inputs_batch}
        if labels_batch is not None:
            feed_dict[self.y]=labels_batch
        return feed_dict

    def make_RNN_cell(self,Nneurons,fn=tf.nn.relu):
        """
        Returns a new cell (for deep recurrent networks), with Nneurons,
        and activation function fn.
        """
        #Make cell type
        if self.cell_type=='basic':
            cell=BasicRNNCell(num_units=Nneurons,activation=fn)
        elif self.cell_type=='LSTM':
            cell=LSTMCell(num_units=Nneurons,activation=fn)
        elif self.cell_type=='GRU':
            cell=GRUCell(num_units=Nneurons,activation=fn)
        #only include dropout when training
        if (self.is_training & self.is_dropout):
            cell=DropoutWrapper(cell,input_keep_prob=self.keep_prob,
                                variational_recurrent=True,
                                input_size=Nneurons,
                                dtype=tf.float32)
        return cell
    
    def add_prediction_op(self):
        """The core model to the graph, that
        transforms the inputs into outputs.
        Implements deep neural network with relu activation.
        """
        ##Tries to make projection to reduce dim from Ninputs to Nhidden
        stacked_inputs=tf.reshape(self.X,[-1,self.Ninputs])
        stacked_inputs=fully_connected(stacked_inputs,self.Nhidden, activation_fn=None)
        inputs_reduced= tf.reshape(stacked_inputs,[-1,self.Nsteps,self.Nhidden])
        #Make multiple cells. Note that using [cell]*n_layers did not work.  This just made a copy pointing at the SAME cell in memory.
        #That led to problems with training. 
        #But calling a function that returns a cell avoids that.
        cell_list=[]
        for i in range(self.Nlayers):
            cell_list.append(self.make_RNN_cell(self.Nhidden,tf.nn.leaky_relu))
        multi_cell=tf.contrib.rnn.MultiRNNCell(cell_list,state_is_tuple=True)
        rnn_outputs,states=tf.nn.dynamic_rnn(multi_cell,inputs_reduced,dtype=tf.float32)
        #use states (like CNN) since need final output state.
        #this maps the number of hidden units back to a different number. 
        outputs = fully_connected(states,self.Noutputs,activation_fn=None)
        outputs=outputs[0]
        return outputs

    def add_loss_op(self,outputs):
        """Add ops for loss to graph.
        Uses mean-square error as the loss.  (Nice, differentiable)
        """
        loss = tf.reduce_mean(tf.square(outputs-self.y))                
        return loss

    def add_training_op(self,loss):
        """Create op for optimizing loss function.
        Can be passed to sess.run() to train the model.
        Return 
        """
        optimizer=tf.train.AdamOptimizer(learning_rate=self.lr)
        training_op=optimizer.minimize(loss)
        return training_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: current tensorflow session
            input_batch:  np.ndarray of shape (Nbatch, Nfeatures)
            labels_batch: np.ndarray of shape (Nbatch, 1)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: current tensorflow session
            input_batch: input data np.ndarray of shape (Nbatch, Nfeatures)
        Returns:
            predictions: np.ndarray of shape (Nbatch, 1)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions


    # #Should use tf.Data as described in seq2seq.
    #Much faster than feed_dict according to TF docs
    
    def get_random_batch(self,X,y):
        """get_random_batch(X,y)   
        Gets multiple random samples for the data.
        Makes list of returned entries.
        Then combines together with 'stack' function at the end.
        Currently selected the next days change in stock price.

        X - matrix of inputs, (Nt, Ninputs)
        Y - matrix of desired outputs (Nt,Noutputs)

        Outputs:
        X_batch - random subset of inputs shape (Nbatch,Nsteps,Ninputs) 
        y_batch - corresponding subset of outputs (Nbatch,Nsteps)
        """
        Nt,Nin = X.shape
        x_list=[]
        y_list=[]
        for i in range(self.Nbatch):
            n0=int(np.random.random()*(Nt-self.Nsteps-1))
            n1 = n0+self.Nsteps
            x_sub = X[n0:n1]
            y_sub = y[n1]
            x_list.append(x_sub)
            y_list.append(y_sub)
        x_batch=np.stack(x_list,axis=0)
        y_batch=np.stack(y_list,axis=0)
        return x_batch,y_batch

    # def get_combined_data(self, ):
    #     """
    #     Try to use the dataset example (following seq2seq tutorial on Tensorflow)
    #     Tensorflow Does all of the splitting, lookup.  
    #     """
    #     # a list of strings.
    #     text_batch = tf.placeholder(tf.string, shape=(self.Nbatch,))
    #     dataset = tf.data.Dataset.from_tensor_slices(text_batch)

    #     label_dataset=tf.Dataset(labels)
        
    #     #Direct from TF
    #     dataset = dataset.map(lambda string: tf.string_split([string]).values)
    #     # dataset = dataset.map(lambda words: (words, tf.size(words)))
    #     #dataset = dataset.map(lambda words, size: (table.lookup(words), size))
    #     dataset=dataset.map(lambda words,size: sentence_lookup(words), tf.size(words))

    #     #zip together
    #     dataset_total=tf.data.Dataset.zip((dataset,label_dataset))

    #     dataset=
    
    def train_graph(self,Xi,yi,save_name=None):
        """train_graph
        Runs the deep NN on the reduced term-frequency matrix.
        """
        self.is_training=True
        #save model and graph
        saver=tf.train.Saver()
        init=tf.global_variables_initializer()
        loss_tot=np.zeros(int(self.n_iter/self.nprint+1))
        #Try adding everything by name to a collection
        tf.add_to_collection('X',self.X)
        tf.add_to_collection('y',self.y)
        tf.add_to_collection('loss',self.loss)
        tf.add_to_collection('pred',self.pred)
        tf.add_to_collection('train',self.train_op)
        
        with tf.Session() as sess:
            init.run()
            # if (save_name!=None):
            #     saver.save(sess,save_name)
            t0=time.time()
            #Use Writer for tensorboard.
            writer=tf.summary.FileWriter("logdir-train",sess.graph)            
            for iteration in range(self.n_iter+1):
                #select random starting point.
                X_batch,y_batch=self.get_random_batch(Xi,yi)
                current_loss=self.train_on_batch(sess, X_batch, y_batch)
                t2_b=time.time()
                if (iteration)%self.nprint ==0:
                    clear_output(wait=True)
                    print('iter #{}. Current MSE:{}'.format(iteration,current_loss))
                    print('Total Time taken:{}'.format(t2_b-t0))
                    print('\n')
                    #save the weights
                    if (save_name != None):
                        saver.save(sess,save_name,global_step=iteration,
                                   write_meta_graph=True)
                    #manual logging of loss    
                    loss_tot[int(iteration/self.nprint)]=current_loss
            writer.close()
            #Manual plotting of loss.  Writer/Tensorboard supercedes this .
            plt.figure()                            
            plt.plot(loss_tot)
            plt.ylabel('Error')
            plt.xlabel('Iterations x{}'.format(self.nprint))
            plt.show()
            

    def predict_all(self,model_name,num,input_data,reset=False):
        """network_predict
        Load a saved Neural network, and predict the output labels
        based on input_data.  Predicts the whole sequence, using
        the batching to process the data in sequence. 
    
        Input: model_name - string name to where model/variables are saved.
        input_data - transformed data of shape (Nobs,Nfeature).

        Output nn_pred_reduced - vector of predicted labels.
        """
        if (reset):
            tf.reset_default_graph()        
        self.is_training=False
        full_model_name=model_name+'-'+str(num)
        with tf.Session() as sess:
            saver=tf.train.import_meta_graph(full_model_name+'.meta')
            #restore graph structure
            self.X=tf.get_collection('X')[0]
            self.y=tf.get_collection('y')[0]
            self.pred=tf.get_collection('pred')[0]
            self.train_op=tf.get_collection('train_op')[0]
            self.loss=tf.get_collection('loss')[0]
            #restores weights etc.
            saver.restore(sess,full_model_name)
            Nin=input_data.shape[0]
            if (Nin < self.Nbatch):
                print('Number of inputs < Number of batch expected')
                print('Padding with zeros')
                input_dat=np.append(input_dat,
                                    np.zeros((self.Nbatch-Nin,self.Noutputs)))
            i0=0
            i1=self.Nbatch
            nn_pred_total=np.zeros((Nin,self.Noutputs))
            while (i1 < Nin-self.Nsteps):
                print(i0,i1)
                #now treat each time, as another element in a batch.
                #(i.e. march through dataset predicting, instead of randomly selecting for training)
                X_batch=np.zeros((self.Nbatch,self.Nsteps,self.Ninputs))
                for i in range(self.Nbatch):
                    X_batch[i,:,:]=input_data[(i0+i):(i0+i+self.Nsteps),:]
                nn_pred=self.predict_on_batch(sess,X_batch)
                sl=slice(self.Nsteps+i0,self.Nsteps+i1)
                nn_pred_total[sl]=nn_pred
                i0=i1
                i1+=self.Nbatch
            #last iter: do remaining operations.  
            Nleft=Nin-i0-self.Nsteps
            X_batch=np.zeros((Nleft,self.Nsteps,self.Ninputs))
            for i in range(Nleft):
                X_batch[i,:,:]=input_data[(i0+i):(i0+i+self.Nsteps),:]
            nn_pred=self.predict_on_batch(sess,X_batch)
            nn_pred_total[-Nleft:]=nn_pred
            #nn_pred_reduced=np.round(nn_pred_total).astype(bool)
        return nn_pred_total

    
    def restore_model(self,sess,model_name,num):
        """Attempts to reset both TF graph, and 
        RNN stored variables/structure.
        """
        saver=tf.train.import_meta_graph(model_name+'.meta')
        #restore graph structure
        self.X=tf.get_collection('X')[0]
        self.y=tf.get_collection('y')[0]
        self.pred=tf.get_collection('pred')[0]
        self.train=tf.get_collection('train')[0]
        self.loss=tf.get_collection('loss')[0]
        #restores weights etc.
        saver.restore(sess,model_name+'-'+str(num))
        
