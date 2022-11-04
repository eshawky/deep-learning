from datetime          import datetime
initial_time    = datetime.now()
print ('Starting time is :  ******************************** ' ,initial_time)
""" **************************************************************************************************************************************** """
import sys
sys.path.append("../")
sys.path.append("../AE/")
sys.path.append("../MFCC/")
sys.path.append("../DFT_FFT/")
sys.path.append("../Bands/")
sys.path.append("../C3D/")
sys.path.append("../FeatureFusion/")

import tensorflow as tf
import numpy as np
from CommonFunctions_RNN      import CommonFunctions_RNN
from Configurations_RNN       import Config_EEG
from Configurations_RNN       import Config_AE_DEAP_Matrix
from Configurations_RNN       import Config_DFT
from Configurations_RNN       import Configurations
from Configurations_RNN       import Config_Model_CNN_3D
from Configurations_RNN       import Config_RNN
from Configurations_RNN       import Config_C3D_RNN
from RNN_Main_Chunks          import RNN_Main
from C3D_SAE_SVM              import SAE
from Configurations_Fusion    import Config_Fusion
from CommonFunctions          import CommonFunctions
from Model_RNN                import Model_RNN
""" *********************************************************************************************************************************** """   
CommonFunctions_RNN = CommonFunctions_RNN()
Config          = Configurations()
config_C3D      = Config_Model_CNN_3D()
Config_EEG      = Config_EEG
RNN_Main        = RNN_Main()
SAE             = SAE()
Config_Fusion   = Config_Fusion()
Config_C3D_RNN  = Config_C3D_RNN()
CommonFunctions = CommonFunctions()
Model_RNN       = Model_RNN()
""" *********************************************************************************************************************************** """   
class C3D_Main(object):
                          
    def main_process (self):
        """ """
        n_users      = Config_EEG.NumOfUsers
        accuracy_file= Config_C3D_RNN.c3d_rnn_accuracy_file
        CompareIndex = 19 #+6
        for userIndex in range (n_users):
            
            with tf.variable_scope('lstm_cells') as scope:
                          
                if userIndex == CompareIndex : #> 14 : # Valence: userIndex = 18, 24, 30, Arousal:userIndex = 5, 11, 17, 19, 25  
                    
                    if Config.OverlapFlag:
                        print ()
                        print ('Sorry Config.OverlapFlag is set True .. ')
                        sys.exit(0)
                        
                    if userIndex > (CompareIndex + 1):
                        scope.reuse_variables()
                    
                    self.prepareRNNData(userIndex, accuracy_file )     

        """ WORKS FINE FOR 2 CLASSES CASE """
        NumOfClasses       = Config_EEG.NumOfClasses 
        AverageAccuracy    = RNN_Main.main_Performance(NumOfClasses, accuracy_file, 4 , plotConfusoin=False)#if svm: 2 else 4 
        print ('Average Accuracy is :', AverageAccuracy)        
        """ """
        
    def prepareRNNData(self, userIndex, accuracy_file):
        
        x                   = Config_C3D_RNN.x
        y                   = Config_C3D_RNN.y
        n_input             = Config_C3D_RNN.n_input
        n_hidden1           = Config_C3D_RNN.num_hidden_nodes1         
        n_hidden2           = Config_C3D_RNN.num_hidden_nodes2         
        NumOfClasses        = Config_C3D_RNN.nclasses 
        n_chunks            = Config_C3D_RNN.n_chunks
        chunk_size          = Config_C3D_RNN.chunk_size
        hm_epochs           = Config_C3D_RNN.hm_epochs
                   
        EEG_train_folds_list, EEG_test_folds_list = SAE.GetModalityFeatures(userIndex ,  Config_Fusion.EEG_SVM_Path )
        Eye_train_folds_list, Eye_test_folds_list = SAE.GetModalityFeatures(userIndex ,  Config_Fusion.Eye_SVM_Path )
        EMG_train_folds_list, EMG_test_folds_list = SAE.GetModalityFeatures(userIndex ,  Config_Fusion.EMG_SVM_Path )
        
        """ Concatenate AE features from 3 Modalities """
        train_folds_list = [EEG_train_folds_list, Eye_train_folds_list, EMG_train_folds_list]
        test_folds_list  = [EEG_test_folds_list , Eye_test_folds_list , EMG_test_folds_list]                

        sum           = 0
        text_file     = open( accuracy_file + str(userIndex + 1) + ".txt", "w")
        
        print ('current user is ******************************************************: ' , userIndex + 1)
        CommonFunctions.makeDirectory(accuracy_file)
        
        for foldId in range(len(EEG_train_folds_list)):    
            
            with tf.variable_scope('lstm_stacked') as scope: 
                  
                    if foldId > 0 :
                        scope.reuse_variables()
                          
                    #if foldId == 0: 
                    
                    print('************************************* Now im in fold ' , foldId+1)
                    FoldsFolder = accuracy_file + 'Folds\\'
                    CommonFunctions.makeDirectory(FoldsFolder)
                    text_file_fold   = open( FoldsFolder + str(userIndex + 1) + '_' + str(foldId + 1) +".txt", "w")
                     
                    prediction    = Model_RNN.Build_RNN(x , n_input,n_hidden1 , n_hidden2, NumOfClasses, n_chunks, chunk_size)
                    cost          = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = Config_RNN.y) )
                    optimizer     = tf.train.RMSPropOptimizer(0.001).minimize(cost)
                    
                    with tf.Session() as sess:
    
                        #Set a seed for python and tensorflow variables
                        np.random.seed    (Config_RNN.WeightSeed)
                        tf.set_random_seed(Config_RNN.WeightSeed)    
                        sess.run( tf.global_variables_initializer() )
                                       
                        train = SAE.concatenateFeatures_C3D (train_folds_list,  foldId) #get all foldId data of train from all modalities
                        test  = SAE.concatenateFeatures_C3D (test_folds_list ,  foldId) #get all foldId data of test from all modalities
                        
                        X_train, Y_train, X_test, Y_test = self.getTrainTestParts(train, test)
                                                          
                        """ train NW """ 
                        Model_RNN.train_NW (sess , optimizer, cost , X_train, Y_train , hm_epochs, n_chunks, chunk_size,x)
      
                        #2-calculate loss
                        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Config_RNN.y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))     
                        Accuracy, fold_confusion_matrix = Model_RNN.test_NW_metrics(sess, accuracy, X_test, Y_test, prediction, n_chunks, chunk_size, x)
                         
                        print ('Accuracy of fold', str (foldId + 1), ' is ' , Accuracy)                       
                            
                        sum     += Accuracy                    
                        text_file_fold.write('Fold ' + str(foldId) +' Validation Accuracy ' + str (Accuracy) + ' %' ) 
                        text_file_fold.close()

        print ('********************************************** foldId now is : ' , str(foldId))                      
        Avg = sum /(foldId+1)
        print ('*************************** Average Fold Validation Accuracy : ' , Avg)
               
        text_file.write('Average Fold Validation Accuracy ' + str (Avg) + ' %' )    
        text_file.close()
             
        return  Avg 

    def getTrainTestParts(self, train, test):
            
            train = SAE.updateAllDataFormat(train)                                
            test  = SAE.updateAllDataFormat(test)

            X_train = train[:,0]
            Y_train = train[:,1]
            X_test  = test [:,0]
            Y_test  = test [:,1]
                        
            X_train_new = []
            Y_train_new = []
            X_test_new  = []
            Y_test_new  = []
            
            for i in range (len(X_train)):        
                data = X_train[i]
                X_train_new.append(data[0])
                
                label = Y_train[i]
                Y_train_new.append(label)
            
            for i in range (len(X_test)):        
                data = X_test[i]
                X_test_new.append(data[0])
                
                label = Y_test[i]
                Y_test_new.append(label)
                
            print ('X_train_new  **: ' , np.array(X_train_new).shape)
            print ('Y_train_new  **: ' , np.array(Y_train_new).shape)
            print ('X_test_new   **: ' , np.array(X_test_new).shape)
            print ('Y_test_new   **: ' , np.array(Y_test_new).shape)

            return X_train_new, Y_train_new, X_test_new, Y_test_new
        
if __name__ == "__main__":
    
    C3D_Main  = C3D_Main()
    C3D_Main.main_process()
 
    Duration = datetime.now() - initial_time
    print ('Duration of system is : ', Duration)

    