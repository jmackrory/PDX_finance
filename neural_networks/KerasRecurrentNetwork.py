#make a Keras model, with RNN with dense input/outputs.
#Borrowing some from Keras docs, and "https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/"
import keras  
from keras.models import Sequential 
from keras.layers import Dense, RNN, LSTM, StackedRNNCells, Reshape, Dropout
from keras.losses import mean_squared_error, mean_absolute_error
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

Netf=7
Nind=10
Nstocks_tot=710

class RNNConfig(object):
    def __init__(self):
        self.Nstocks=1
        self.Nfeatures=1
        #number of times in/out.
        self.Ntime_in=20
        self.Ntime_out=10
        #total linear input/outputs
        self.Ninput=self.calc_Ninput()
        self.Noutput=Netf
        #network parameters
        self.Nhidden=20
        self.Nlayers=2
        self.dropout_frac=0.5
        self.lr=0.001
        self.Nepoch=100
        self.Nbatch=100
        self.Nprint=10
        self.train_frac=0.75

    def calc_Ninput(self):
        y=self.Nstocks*self.Nfeatures+Netf+Nind
        return y

class KerasRecurrentNetwork(object):
    """
    Module to create and train a Recurrent Network in Keras.

    Contains:
    make_model
    train_model
    avg_predict_from_model
    get_batch
    get_training_data
    """
    Netf=7
    Netf=10
    def __init__(self,config):
        """
        Wrapper for Keras Network.  Uses multiple layers.
        Reshapes values from multiple times to a single vector.
        Network outputs single vector of ETFs as multiple times.
        Accepts config object which gets passed on to everything else.  
        """
        self.conf=config
        keras.backend.clear_session()
        self.model=Sequential()

    #make a Keras model two layer LSTM, with dense input/outputs
    #Use the sequence-to-sequence example.  
    def make_network(self,activ='relu'):
        """make_network
        Actually create deep RNN, with chosen activation.
        Uses dense layers before/after RNN to reduce dimensionality.

        """
        keras.backend.clear_session()
        self.model=Sequential()
        self.model.add(Dense(units=self.conf.Nhidden, activation='linear',
                        input_shape=(self.conf.Ntime_in,self.conf.Ninput))) #linear mapping at input
        if (activ=='relu'):
            act = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)               
            for n in range(self.conf.Nlayers):
                self.model.add(Dropout(rate=self.conf.dropout_frac,
                                  noise_shape=(self.conf.Nbatch,1,self.conf.Nhidden))) #dropout at input                
                self.model.add(LSTM(units=self.conf.Nhidden,
                               activation='linear',return_sequences=True))
                self.model.add(act)            #add extra activation layer afterwards
            #final LSTM layer
            # self.model.add(Dropout(rate=self.conf.dropout_frac,
            #                   noise_shape=(self.conf.Nbatch,1,self.conf.Nhidden))) #dropout at input                            
            # self.model.add(LSTM(units=self.conf.Nhidden,
            #                activation='linear',return_sequences=True))
            # self.model.add(act)
        else:
            for n in range(self.conf.Nlayers):
                #dropout at input                                        
                self.model.add(Dropout(rate=self.conf.dropout_frac,
                                  noise_shape=(self.conf.Nbatch,1,self.conf.Nhidden))) 
                self.model.add(LSTM(units=self.conf.Nhidden,
                               activation=activ,return_sequences=True))
            # self.model.add(Dropout(rate=self.conf.dropout_frac,
            #                   noise_shape=(self.conf.Nbatch,1,self.conf.Nhidden))) 
            # self.model.add(LSTM(units=self.conf.Nhidden,activation=activ,return_sequences=True))
            
        #Now flatten that sequence of outputs down.           
        self.model.add(Reshape(input_shape=(self.conf.Ntime_in,self.conf.Nhidden),
                          target_shape=(self.conf.Ntime_in*self.conf.Nhidden,)))
        #linear mapping to change dimension at output
        self.model.add(Dense(units=self.conf.Noutput*self.conf.Ntime_out, activation='linear',
                        input_shape=(self.conf.Ntime_in*self.conf.Nhidden,))) 
        #reshape again that sequence down.           
        self.model.add(Reshape(input_shape=(self.conf.Noutput,),
                          target_shape=(self.conf.Ntime_out,self.conf.Noutput)))
        adam=keras.optimizers.adam(lr=self.conf.lr, clipnorm=1)
        self.model.compile(optimizer=adam,loss=mean_squared_error)


    def get_training_data(self,X):
        """get_training_data
        Selects out a subset of the training data.
        Requires monkey around with column indices as input data is of form:
        [ ...stocks..., ETFS, Indicators ]
        Those last two are static, and known.
        Picks out a fraction of the input data and trains the rest on that. 
        """
        #select out desired range of columns for training (stocks, etf, ind)
        # indx0=np.arange(self.conf.Nstocks)
        # Nrow,Ncol=X.shape
        
        # ind_etf=np.arange(Netf)
        # ind_x=np.arange(Nind+Netf)        
        # if (self.conf.Nstocks>0):
        #     indx0=np.arange(self.conf.Nstocks)
        #     ind_x=np.append(ind_x,indx0)
        #     for i in range(self.conf.Nfeatures-1):
        #         ind_x=np.append(i*Nstocks_tot+indx0,ind_x)
        Nrow,Ncol=X.shape        
        Nc=int(self.conf.train_frac*Nrow)
        self.get_training_indices(X)
        
        #make training/test splits
        #train on stock, indicators and ETFs.
        Xtrain = X[:Nc,self.ind_x]
        ytrain = X[:Nc,self.ind_y]
        return Xtrain,ytrain

    def get_training_indices(self,X):
        """get_training_data
        Selects out a subset of the training data.
        Requires monkey around with column indices as input data is of form:
        [ ...stocks..., ETFS, Indicators ]
        Those last two are static, and known.
        Picks out a fraction of the input data and trains the rest on that. 
        """
        #select out desired range of columns for training (stocks, etf, ind)
        indx0=np.arange(self.conf.Nstocks)
        Nrow,Ncol=X.shape
        ind_etf=np.arange(Netf)
        ind_x=np.arange(Nind+Netf)        
        if (self.conf.Nstocks>0):
            indx0=np.arange(self.conf.Nstocks)
            ind_x=np.append(ind_x,indx0)
            for i in range(self.conf.Nfeatures-1):
                ind_x=np.append(i*Nstocks_tot+indx0,ind_x)
        self.ind_x = ind_x
        self.ind_y = ind_etf

    
    def get_batch(self,X,y):
        """get_batch
        Returns a randomly selected batch of input/output sequences.
        Inputs are all stocks, ETFs and indicators
        Outputs are just future ETFs from input sequence endpoint.
        """
        #starting indices
        ind=np.arange(len(X[:,0])-self.conf.Ntime_in-self.conf.Ntime_out)
        rand_ind=np.random.choice(ind,self.conf.Nbatch,replace=False)
        Xb=np.zeros((self.conf.Nbatch,self.conf.Ntime_in,self.conf.Ninput))
        yb=np.zeros((self.conf.Nbatch,self.conf.Ntime_out,self.conf.Noutput))
        #now populate table (couldn't see nice way to vectorize this assignment, mabe via overloading)
        for i in range(self.conf.Nbatch):
            t0=rand_ind[i]
            t1=t0+self.conf.Ntime_in
            t2=t1+self.conf.Ntime_out
            #input all past parameters
            Xb[i]=X[t0:t1]
            #target future ETFs
            yb[i]=y[t1:t2]
        return Xb,yb,rand_ind

    def train_model(self,Xtrain,ytrain):
        """train_model
        Grabs random sub-batches of data, then trains.
        Uses two different calls to suppress output.
        """
        for i in range(self.conf.Nepoch+1):
            #Keras assumes you have a list of X,y pairs for its sampling.
            #Would be memory intensive to set up a whole list for this data.
            #So wrote my own batching.
            #Should really finish Batch generator.
            Xb,yb,_=self.get_batch(Xtrain,ytrain)
            if (i)%self.conf.Nprint==0:
                self.model.fit(Xb,yb, epochs=1, batch_size=self.conf.Nbatch, verbose=1)
            else:
                self.model.fit(Xb,yb, epochs=1, batch_size=self.conf.Nbatch, verbose=0)
            self.model.reset_states()

    def avg_predict_from_model(self,X):
        """avg_predict_from_model

        Currently runs all prediction on the given input X.  
        Also currently takes a naive AVERAGE(!) over all of the output predictions. 
        """
        #Predict on whole of this subset (both "training" and "testing")

        #compute total number of predictions to be made. 
        Nf = len(X) - self.conf.Ntime_in - self.conf.Ntime_out
        ypred_tot=np.zeros((len(X),Netf))
        yavg = np.zeros((len(X),1))
        i0=0
        i1=i0+self.conf.Nbatch
        #split whole time sequence into sequential batches.
        while (i1 < Nf):
            X0=np.zeros((self.conf.Nbatch,self.conf.Ntime_in,self.conf.Ninput))
            for i in range(self.conf.Nbatch):
                t0=i0+i
                t1=t0+self.conf.Ntime_in
                X0[i]=X[t0:t1,self.ind_x]
            ypred=self.model.predict(X0,batch_size=self.conf.Nbatch)
            #now march along batch, add up predictions.    
            for i in range(self.conf.Nbatch):
                t0=self.conf.Ntime_in+i0+i
                t1=t0+self.conf.Ntime_out
                yi=ypred[i].reshape( (self.conf.Ntime_out, Netf))
                ypred_tot[t0:t1]+= yi
                yavg[t0:t1]+=1
            self.model.reset_states()    
            i0=i1
            i1+=self.conf.Nbatch
        #predict on the remainder
        Nrem=Nf-i0
        X0=np.zeros((Nrem,self.conf.Ntime_in,self.conf.Ninput))
        for i in range(Nrem):
            t0=i0+i
            t1=t0+self.conf.Ntime_in
            X0[i]=X[t0:t1,self.ind_x]
        ypred=self.model.predict(X0,batch_size=Nrem)
        #now march along batch, add up predictions.    
        for i in range(Nrem):
            t0=self.conf.Ntime_in+i0+i
            t1=t0+self.conf.Ntime_out
            yi=ypred[i].reshape((self.conf.Ntime_out,Netf))
            ypred_tot[t0:t1]+= yi
            yavg[t0:t1]+=1

        ypred_tot=ypred_tot/yavg    

        return ypred_tot
