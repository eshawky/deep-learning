from datetime          import datetime
initial_time    = datetime.now()
print ('Starting time of Cross Validation is  ****************** ' ,initial_time)
""" **************************************************************************************************************************************** """
import sys
sys.path.append("../")
sys.path.append("../AE/")
sys.path.append("../MFCC/")
sys.path.append("../DFT_FFT/")
sys.path.append("../Bands/")
sys.path.append("../RNN/")

import numpy as np
import os
import tensorflow as tf
from   sklearn.model_selection import KFold
#from tensorflow.python.ops import variable_scope as vs
#from sklearn import svm
""" ************************************************************************************************************************************* """   
from CommonFunctions          import CommonFunctions
from CommonFunctions_EEG      import CommonFunctions_EEG
from Configurations_AE        import Configurations
from Configurations_AE        import Config_Model_CNN_3D
from Configurations_AE        import Config_AE_DEAP_Matrix
from Configurations_AE        import Config_EEG
from Configurations_AE        import Config_Without
from Configurations_AE        import Config_DFT
from Configurations_AE        import Config_MFCC
from PerformanceMetrics_Multi import PerformanceMetrics_Multi
from Model_CNN_3D_DFT         import CNN_3D  
from sklearn                  import datasets
from sklearn.multiclass       import OneVsOneClassifier
from sklearn.svm              import LinearSVC
from DFT                      import DFT

from Preprocess_RNN           import Preprocess_RNN
from Configurations_AE        import Config_RNN
from Model_RNN                import Model_RNN
""" *********************************************************************************************************************************** """   
CNN_3D             = CNN_3D()    
CommonFunctions    = CommonFunctions()
CommonFunctions_EEG=CommonFunctions_EEG()
pm                 = PerformanceMetrics_Multi()
Config             = Configurations()
Config_AE          = Config_AE_DEAP_Matrix()
config_C3D         = Config_Model_CNN_3D()
Config_EEG         = Config_EEG
Config_Without     = Config_Without()
DFT                = DFT()
Config_DFT         = Config_DFT()
Config_MFCC        = Config_MFCC()

Config_RNN         = Config_RNN()
Model_RNN          = Model_RNN()
Preprocess_RNN     = Preprocess_RNN()
""" *********************************************************************************************************************************** """   
class RNN_Main(object):
    
    def main_process(self, userIndex):
        """ """
        #Preprocess_RNN.main_process(userIndex)
        #self.main_process_random(userIndex)
        self.main_process_kfold (userIndex)
        
    def main_process_kfold(self, userIndex ):

        currentLablePath    = Config_RNN.currentWorkingLabel #say valence
        NumOfClasses        = Config_RNN.nclasses 
        hm_epochs           = Config_RNN.hm_epochs
        x                   = Config_RNN.x
        y                   = Config_RNN.y        
        
        data_path           = Config_RNN.data_file   + str(userIndex + 1) + '.csv'
        label_path          = Config_RNN.labels_file + str(userIndex + 1) + '.csv'
        rnn_accuracy_file   = Config_RNN.rnn_accuracy_file  
        n_input             = Config_RNN.chunk_size
        n_hidden1           = Config_RNN.num_hidden_nodes1         
                
        if Config.currentWorkingLabel == 'valence' or Config.currentWorkingLabel == 'arousal':
                        
            Avg = self.CrossValidation (data_path , label_path , userIndex,  rnn_accuracy_file, n_input, n_hidden1, x, NumOfClasses, hm_epochs)            
    
    def main_process_random(self , userIndex):
        """ 
        Apply Random Split and RNN for EEG Data
        """
        x                   = Config_RNN.x
        y                   = Config_RNN.y
        
        np.random.seed (Config_RNN.WeightSeed)
        tf.set_random_seed(Config_RNN.WeightSeed)
 
        print ('SPLITTING CHUNKS INTO TRAIN AND TEST ...')
        train_test_file_path             = Config_RNN.data_file   + str(userIndex + 1) + '.csv'
        LablesPath                       = Config_RNN.labels_file + str(userIndex + 1) + '.csv'
        
        n_input             = Config_RNN.chunk_size
        n_hidden1           = Config_RNN.num_hidden_nodes1         
         
        X_train, X_test, Y_train, Y_test = Model_RNN.splitIntoTrainTest(train_test_file_path, LablesPath,split_ratio = Config_RNN.split_ratio, split_random_state=Config_RNN.split_random_state)
        X_train_new , Y_train_new , X_test_new , Y_test_new = Model_RNN.update_splitIntoTrainTest(X_train, X_test, Y_train, Y_test)
 
        prediction          = Model_RNN.Build_RNN (x, n_input, n_hidden1 , Config_RNN.nclasses ) 
        FinalAccuracy       = Model_RNN.Train_Test_Random (x , y , X_train_new , Y_train_new , X_test_new , Y_test_new , prediction)
    
    """ *********************************************************************************************************************************** """                                  
    def CrossValidation (self, user_path , label_path , userIndex , accuracy_file, n_input, n_hidden1,x, NumOfClasses, hm_epochs):
            
        text_file = open( accuracy_file + str(userIndex + 1) + ".txt", "w")
        sum       = 0 
        Avg       = 0
        k         = 0
        counter   = 10
        all_folds_confusion_matrix = []
                
        # #Read full data
        AllData = CommonFunctions.Get_from_csv_Array(user_path)       
        print ('AllData.shape' , AllData.shape)
          
        #Read valence labels (one hot)
        AllLabels = CommonFunctions.Get_from_csv_Array(label_path)       
        print ('AllLabels.shape' , AllLabels.shape)
        
        kf          = KFold(n_splits = Config.Kfold, shuffle=True , random_state=Config_RNN.WeightSeed)
        kf.get_n_splits(AllData)
                
        #for train_index, validate_index in kf.split(AllData):
                
        #if k == 0 :
        #counter = 0 #if k==0 set counter = 0
        
        #print('******************************************************************************************* Now im in fold ' , k+1)
        #FoldsFolder = accuracy_file + 'Folds\\'
        #CommonFunctions.makeDirectory(FoldsFolder)
        #text_file_fold   = open( FoldsFolder + str(userIndex + 1) + '_' + str(k+1) +".txt", "w")
        
        #prediction    = Model_RNN.Build_RNN(x , n_input,n_hidden1 , NumOfClasses)
        #cost          = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = Config_RNN.y) )
        #optimizer     = tf.train.AdamOptimizer().minimize(cost)
        
        with tf.Session() as sess:
                    
#            prediction    = Model_RNN.Build_RNN(x , n_input,n_hidden1 , NumOfClasses)
#            cost          = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = Config_RNN.y) )
#            optimizer     = tf.train.AdamOptimizer().minimize(cost)
           
            with tf.variable_scope('lstm_stacked') as scope:
                
               for train_index, validate_index in kf.split(AllData):
                        
                        if k > 0 :
                            scope.reuse_variables()
                            
                        #counter = 0 #if k==0 set counter = 0
                        
                        print('******************************************************************************************* Now im in fold ' , k+1)
                        FoldsFolder = accuracy_file + 'Folds\\'
                        CommonFunctions.makeDirectory(FoldsFolder)
                        text_file_fold   = open( FoldsFolder + str(userIndex + 1) + '_' + str(k+1) +".txt", "w")
                        
                        #prediction    = Model_RNN.Build_RNN(x , n_input,n_hidden1 , NumOfClasses)
                        #cost          = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = Config_RNN.y) )
                        #optimizer     = tf.train.AdamOptimizer().minimize(cost)                   
                        
                        #Set a seed for python and tensorflow variables
                        np.random.seed    (Config_RNN.WeightSeed)
                        tf.set_random_seed(Config_RNN.WeightSeed)    
                        sess.run( tf.global_variables_initializer() )
                        
                        #X_train, Y_train, X_validate, Y_validate = CommonFunctions_EEG.get_train_validate_partitions(AllData, train_index, validate_index)
                        X_train, Y_train, X_validate, Y_validate = self.readRNNDataLabels(AllData , AllLabels , train_index, validate_index)
                        
                        """ train NW """ 
                        Model_RNN.train_NW (sess , optimizer, cost , X_train, Y_train , hm_epochs)
                        
                        #2-calculate loss
                        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Config_RNN.y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))     
                        Accuracy, fold_confusion_matrix = Model_RNN.test_NW_metrics(sess, accuracy, X_validate, Y_validate, prediction)
                        
                        print ('Accuracy of fold', str (k + 1), ' is ' , Accuracy)                       
                        
                        sum     += Accuracy                    
                        all_folds_confusion_matrix.append(fold_confusion_matrix)
                        text_file_fold.write('Fold ' + str(k) +' Validation Accuracy ' + str (Accuracy) + ' %' ) 
                        text_file_fold.close()
                        
                        k += 1

        print ('********************************************** k now is : ' , str(k))                  
        if counter == 0:
            Avg = sum 
        else:
            Avg = sum /(k)
        print ('************************* Average Fold Validation Accuracy ' , Avg)       
        text_file.write('Average Fold Validation Accuracy ' + str (Avg) + ' %' )    
        user_confusion_matrix = pm.getUserConfusionMatrix(all_folds_confusion_matrix, NumOfClasses)
        text_file.write( '\n user_confusion_matrix       ***** \n' + str (user_confusion_matrix))
        text_file.close()
            
        return  Avg 
    
    def readRNNDataLabels(self,train_test , labels , trainIndex, testIndex):
        """ """

        X_train = train_test[trainIndex]
        Y_train = labels    [trainIndex]
        
        X_test  = train_test[testIndex]        
        Y_test  = labels    [testIndex]
        
        X_train , Y_train ,X_test,Y_test = Model_RNN.update_splitIntoTrainTest(X_train, X_test, Y_train, Y_test)

        return X_train , Y_train , X_test , Y_test 
                       
if __name__ == "__main__":
    
    userIndex = 0
    print (' ******************************************************** current user is : ' , str(userIndex+1) )
    
    RNN_Main  = RNN_Main()
    RNN_Main.main_process (userIndex)
    
    end       = datetime.now()
    Duration  = end - initial_time
    print ('*******************Duration of this system is ' ,Duration)      
    
    
    
    
    
    