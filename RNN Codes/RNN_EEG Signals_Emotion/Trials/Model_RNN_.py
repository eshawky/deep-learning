from datetime          import datetime
initial_time           = datetime.now()
print ('Starting time is :  ******************************** ' ,initial_time)
""" **************************************************************************************************************************************** """
import sys
sys.path.append("../")
import csv
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from CommonFunctions       import CommonFunctions
from Configurations_AE     import Config_RNN
from PerformanceMetrics_Multi import PerformanceMetrics_Multi

CommonFunctions  = CommonFunctions()
Config_RNN       = Config_RNN()
pm               = PerformanceMetrics_Multi()
""" ************************************************************************ """
class Model_RNN:    
        
    """ ******************************************************************************************************************************** """       
    def Build_RNN(self, x, n_input , n_hidden1 , n_hidden2, numOfClasses):
        """ """
        #tf.reset_default_graph() 
        x = tf.transpose(x , [1 , 0 , 2] )
        x = tf.reshape  (x , [-1, Config_RNN.chunk_size] )
        x = tf.split    (0 , Config_RNN.n_chunks , x ) 
            
        lstm_last_output = self.RNN_HiddenLayers(x , n_input, n_hidden1, n_hidden2)
        prediction       = self.RNN_DenseLayer(lstm_last_output , n_hidden1, numOfClasses)

        return prediction

    def RNN_HiddenLayers(self , x , n_input, n_hidden1, n_hidden2 , name='Hidden'):
        
            x = self.RNN_HiddenLayer(x , n_input, n_hidden1 , name='layer1')
            x = self.RNN_HiddenLayer(x , n_hidden1, n_hidden2 , name='layer2')

            lstm_cell_1   = rnn_cell.LSTMCell(n_hidden1 , state_is_tuple = True , forget_bias = 1.0 )
            lstm_cell_2   = rnn_cell.LSTMCell(n_hidden2 , state_is_tuple = True , forget_bias = 1.0 )

            #lstm_cell_1    = rnn_cell.LSTMCell(n_hidden1 , state_is_tuple = True , forget_bias = 1.0 )
            #lstm_cell_2    = rnn_cell.LSTMCell(n_hidden2 , state_is_tuple = True , forget_bias = 1.0 )  
            lstm_cells     = rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2] , state_is_tuple = True )  
            outputs, states= rnn.rnn(lstm_cells , x , dtype = tf.float32)             
            
            lstm_last_output= outputs[-1]    
        
            return lstm_last_output
    
    
    def RNN_HiddenLayer(self , x , n_input, n_output , name='Hidden'):
    
        with tf.name_scope(name):
                
                w_hidden    = tf.Variable(tf.truncated_normal( [n_input    , n_output]      , seed = Config_RNN.WeightSeed), name='W')
                b_hidden    = tf.Variable(tf.truncated_normal( [n_output]     , seed = Config_RNN.WeightSeed) , name='B')
                    
                x = tf.nn.relu(tf.matmul( x , w_hidden) + b_hidden)

                return x
                
    def RNN_DenseLayer(self , lstm_last_output , n_hidden1, numOfClasses, name='Dense'):
        """ """
        with tf.name_scope(name):
            w_out      = tf.Variable(tf.truncated_normal ( [n_hidden1 , numOfClasses] , seed = Config_RNN.WeightSeed) , name='W')
            b_out      = tf.Variable(tf.truncated_normal( [numOfClasses]  , seed = Config_RNN.WeightSeed) , name='B')
            
            prediction =  tf.matmul ( lstm_last_output, w_out) + b_out
        
            return prediction

    """ ******************************************************************************************************************************** """       
    def train_NW(self, sess, optimizer, cost,X_train, Y_train, hm_epochs):
        
        zero_loss_count   = 0 #flag for zero loss for early stopping
        for epoch in range(hm_epochs):
                 
            epoch_loss  = 0
            index       = 0
            
            """ If batch size is not divisable with data size"""   
            numOfBatchs= len(X_train)/Config_RNN.batch_size
            if isinstance(numOfBatchs, int) == False:   #batch size not divisable by data
                numOfBatchs = int(numOfBatchs)
                numOfBatchs = numOfBatchs + 1
            
            for xx in range( numOfBatchs):                                                
                 
                #epoch_x , epoch_y  = self.getNextBatch(X_train,Y_train,  numOfBatchs, index , xx, Config_RNN.batch_size)                 
                if xx < (numOfBatchs-1):
                    epoch_x , epoch_y,_  = self.getNextBatch(X_train,Y_train,  numOfBatchs, index , xx, Config_RNN.batch_size)
                    #print ('epoch_x.shape ' , epoch_x.shape)
                    #print ('epoch_y.shape ' , epoch_y.shape)
                    epoch_x              = epoch_x.reshape( (Config_RNN.batch_size, Config_RNN.n_chunks , Config_RNN.chunk_size ))            
                else:
                    epoch_x , epoch_y,lastSamples  = self.getNextBatch(X_train,Y_train,  numOfBatchs, index , xx, Config_RNN.batch_size)                            
                    #print ('epoch_x.shape ' , epoch_x.shape)
                    #print ('epoch_y.shape ' , epoch_y.shape)
                    epoch_x                        = epoch_x.reshape( (lastSamples, Config_RNN.n_chunks , Config_RNN.chunk_size ))
                     
                _, c = sess.run ( [optimizer, cost], feed_dict = {Config_RNN.x: epoch_x, Config_RNN.y: epoch_y}) 

                epoch_loss   += c
                index        += Config_RNN.batch_size

            #Check for Early stopping
            if epoch_loss == 0 :
                zero_loss_count += 1                  
            if zero_loss_count > Config_RNN.zero_loss_ratio:
                print ('No improvement found so Stop optimization ... ')
                break
             
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss :', epoch_loss )
    
    def test_NW_metrics(self,sess, accuracy, X_test, Y_test, prediction):
        """ Formal function for testing using 3D-CNN """
        index         = 0 
        FinalAccuracy = 0
        
        """ If batch size is not divisable with data size"""    
        #numOfBatchs= int (len(X_test)/config_C3D.train_batch_size)
        numOfBatchs= len(X_test)/Config_RNN.batch_size
        if isinstance(numOfBatchs, int) == False: #batch size not divisable by data
            numOfBatchs = int(numOfBatchs)
            numOfBatchs = numOfBatchs + 1
        #print ('numOfBatchs ' , numOfBatchs)
        #print ('len (X_test) ************************ ' , len(X_test))

        fold_confusion_matrix =[ ]           
        for xx in range(numOfBatchs):
                            
            #epoch_x , epoch_y  = self.getNextBatch ( X_test , Y_test,  numOfBatchs, index , xx, Config_RNN.batch_size)
            if xx < (numOfBatchs-1):
                epoch_x , epoch_y,_  = self.getNextBatch(X_test , Y_test,  numOfBatchs, index , xx, Config_RNN.batch_size)
                #print ('epoch_x.shape ' , epoch_x.shape)
                #print ('epoch_y.shape ' , epoch_y.shape)
                epoch_x              = epoch_x.reshape( (Config_RNN.batch_size, Config_RNN.n_chunks , Config_RNN.chunk_size ))                              
            else:
                epoch_x , epoch_y,lastSamples  = self.getNextBatch(X_test,Y_test,  numOfBatchs, index , xx, Config_RNN.batch_size)
                #print ('epoch_x.shape ' , epoch_x.shape)
                #print ('epoch_y.shape ' , epoch_y.shape)  
                epoch_x                        = epoch_x.reshape( (lastSamples, Config_RNN.n_chunks , Config_RNN.chunk_size ))
                                      
            #metrics
            y_p = tf.argmax(prediction, 1)
            Accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={Config_RNN.x:[i for i in epoch_x], Config_RNN.y:[i for i in epoch_y]})
            #print ("y_pred:", y_pred)
            
            y_true = np.argmax(epoch_y,1)
            
            batch_confusion_matrix = pm.getConfusionMatrix (y_true , y_pred)
            fold_confusion_matrix.append(batch_confusion_matrix)
              
            #print ('Current Accuracy is ' , Accuracy * 100, ' %' ) 
            index         += Config_RNN.batch_size
            FinalAccuracy += Accuracy        

        mean_batches_accuracy = (FinalAccuracy/numOfBatchs )
        return mean_batches_accuracy* 100  , fold_confusion_matrix
    
    def Train_Test_Random(self, x , y , X_train , Y_train , X_test , Y_test , prediction):
        
        print ('*************************************** Im training RNN now ... ***************************************')
           
        cost      = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
        optimizer = tf.train.AdamOptimizer(0.0025).minimize(cost)
        
#         cost      = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, targets = y) )                
#         optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)
        
        with tf.Session() as sess:
             
            sess.run(tf.global_variables_initializer())
            
            """ If batch size is not divisable with data size"""   
            numOfBatchs= len(X_train)/Config_RNN.batch_size
            if isinstance(numOfBatchs, int) == False:   #batch size not divisable by data
                numOfBatchs = int(numOfBatchs)
                numOfBatchs = numOfBatchs + 1
            
            #numOfBatchs = int (len(X_train)/self.batch_size)
            print ('numOfBatchs : ' , numOfBatchs)
            
            """START TRAINING ... """
            zero_loss_count   = 0               #flag for zero loss for early stopping
            for epoch in range(Config_RNN.hm_epochs):
                
                epoch_loss = 0            
                index      = 0
                
                for xx in range(numOfBatchs ):              
                    print ('Processing batch ' , str(xx+1) , ' out ot ' , str(numOfBatchs))
                    
                    if xx < (numOfBatchs-1):
                        epoch_x , epoch_y,_  = self.getNextBatch(X_train,Y_train,  numOfBatchs, index , xx, Config_RNN.batch_size)
                        epoch_x = epoch_x.reshape( (Config_RNN.batch_size, Config_RNN.n_chunks , Config_RNN.chunk_size ))
                    
                    else:
                        epoch_x , epoch_y,lastSamples  = self.getNextBatch(X_train,Y_train,  numOfBatchs, index , xx, Config_RNN.batch_size)
                        epoch_x = epoch_x.reshape( (lastSamples, Config_RNN.n_chunks , Config_RNN.chunk_size ))
                        
                    _, c        = sess.run(   [optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}   )
                    epoch_loss += c                
                    index      += Config_RNN.batch_size
    
                #Check for Early stopping
                if epoch_loss == 0 :
                    zero_loss_count += 1                  
                if zero_loss_count > Config_RNN.zero_loss_ratio:
                    print ('No improvement found so Stop optimization  ')
                    break
                
                print('Epoch ', epoch + 1, 'completed out of ' , Config_RNN.hm_epochs,'loss: ',int (epoch_loss) )
                    
            """START TESTING ... """
            correct  = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
            
            print('Accuracy of test: ' , accuracy.eval({x : X_test.reshape((-1 , Config_RNN.n_chunks , Config_RNN.chunk_size )) , y : Y_test }) * 100)
         
    """ ******************************************************************************************************************************** """ 
    #                      (X_test  , Y_test,  numOfBatchs, index , xx, Config_RNN.batch_size) 
    def getNextBatch (self, X_train , Y_train,  numOfBatchs, index , xx, batch_size):
        """ """
        #print ('Training batch ' + str (xx+1) +' out of '+ str (numOfBatchs) +' batches')  
        lastSamples = -1
        if xx == numOfBatchs-1:                   
            #Get last train samples in data and labels to train model with
            #print ('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            
            lastSamples = len(X_train) % batch_size
            epoch_x = np.array ( X_train[-lastSamples:] )
            epoch_y = np.array ( Y_train[-lastSamples:] )
            #print('last samples ' , lastSamples)
        else:
            #print ('Heraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            epoch_x = np.array ( self.my_next_batch ( X_train  , batch_size , index) )
            epoch_y = np.array ( self.my_next_batch ( Y_train  , batch_size , index) )
    
        #update batch type t float32
        epoch_x  = epoch_x.astype(np.float32) 
        epoch_y  = epoch_y.astype(np.float32) 
        
        return epoch_x , epoch_y, lastSamples
    
    def my_next_batch(self, matrix , batch_size , index):
    
        idx = index + batch_size
        matrix = np.array(matrix)
        
        if idx > matrix.shape[0]:
            batch = matrix[index:] #get remaining samples even if they are of size of batch_size
        else:         
            batch = matrix[index : index + batch_size]
        return batch
    
    """ ******************************************************************************************************************************** """ 
    def splitIntoTrainTest(self, train_test_file_path, label_OneHot_csv_path,split_ratio, split_random_state):
    
        from sklearn.model_selection import train_test_split
    
        """ 
        split data into train and test according to split ratio
        """
    
        # #Read full data
        train_test = CommonFunctions.Get_from_csv_Array(train_test_file_path)       
        print ('train_test.shape' , train_test.shape)
          
        #Read valence labels (one hot)
        labels = CommonFunctions.Get_from_csv_Array(label_OneHot_csv_path)       
        print ('labels.shape' , labels.shape)
        
        #labels = labels [:,3:5] #Deleting header :valence word        
                 
        #Split data and labels according to given ratio
        X_train, X_test, Y_train, Y_test = train_test_split(train_test, labels, test_size= split_ratio, random_state =split_random_state)          
    
        return X_train, X_test, Y_train, Y_test 

    def update_splitIntoTrainTest(self, X_train, X_test, Y_train, Y_test):
        """ 
        Loop Through all chunks and update chunks and labels 
        since they are stored in format strings
        such as chunk has ['[0.1' , '0.4', ... , '0.9]' ] 
        we need to delete these [ and ] from the start and end of chunk and also labels
        """
         
        X_train_new = []
        Y_train_new = []
        X_test_new  = []
        Y_test_new  = []
                 
        for i in range (len(X_train)):
            X_train_new.append(self.process_chunk(X_train[i]) )
            Y_train_new.append(self.process_label2(Y_train[i]) )
         
        for i in range (len(X_test)):    
            X_test_new.append(self.process_chunk(X_test[i]) )
            Y_test_new.append(self.process_label2(Y_test[i]) )
         
        X_train_new = np.array(X_train_new)
        Y_train_new = np.array(Y_train_new)
        X_test_new  = np.array(X_test_new)
        Y_test_new  = np.array(Y_test_new)
        
        print ('train  : ' , X_train.shape)
        print ('train  : ' , Y_train.shape)
        print ('test   : ' , X_test.shape)
        print ('test   : ' , Y_test.shape)
        
        return X_train_new , Y_train_new , X_test_new , Y_test_new

    def process_chunk(self , chunk):
        """ 
        Each chunk has ['[0.1' , '0.4', ... , '0.9]' ] 
        We need to delete these [ and ] from the start and end of chunk
        """
        sample1              = chunk[0] 
        sample2              = chunk[-1]
        
        sample1 = sample1.split('[')
        sample2 = sample2.split(']')
        
        chunk[0]  = sample1[1]
        chunk[-1] = sample2[0]
        sample1               = chunk[0] 
        sample2               = chunk[-1]
    
        return chunk
    
    def process_label1(self, label):
        """ 
        Each label has ['[0 1]' ] 
        We need to delete these [ and ] from the start and end of labels
        """
        sample1   = label[0] 
        #print ('sample1 ' , sample1)
    
        sample1   = sample1.split('[')
        #print ('sample1 ' , sample1)
    
        sample1   = sample1[1].split(']')
        #print ('sample1 ' , sample1)
    
        new_label  = [sample1[0][0] , sample1[0][2]]
        return new_label
    
    def process_label2(self, label):
        """ 
        Each label has [ '[0'   ,  '1]' ] 
        We need to delete these [ and ] from the start and end of labels
        """
        #print ('old label : ' , label)
        
        sample1   = label[0] #[0
        #print ('sample1 ' , sample1)
    
        sample1   = sample1.split('[')
        #print ('sample1 ' , sample1)
    
        sample2 = label[1] 
        #print ('sample2 ' , sample2)
    
        sample2 = sample2.split(']')
        #print ('sample2 ' , sample2)
    
        new_label  = [sample1[1] , sample2[0]]
        #print ('new label : ' , new_label)
        
        return new_label
    