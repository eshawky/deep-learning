import os
import csv
import numpy  as np
import pandas as pd

from Model_RNN           import Model_RNN
from CommonFunctions     import CommonFunctions
from Configurations_RNN  import Config_RNN
from Configurations_RNN  import Config_DFT

Model_RNN        = Model_RNN()
CommonFunctions  = CommonFunctions()
Config_RNN       = Config_RNN()
Config_DFT       = Config_DFT()
""" ************************************************************************ """            
class Preprocess_RNN():
    
    def main_process(self, userIndex):
    
        """READING CHUNKS """
        rnn_path      = Config_RNN.rnn_path
        chunks_path   = Config_RNN.chunks_path # Config_DFT.MFCC_Raw_Frames #Config_RNN.chunks_path  
        print ('current chunks path : ********************** ', chunks_path)
          
        currentLabel  = Config_RNN.currentWorkingLabel
        
        data_file     = rnn_path + currentLabel + '\\RNN_Chunks\\' 
        labels_file   = rnn_path + currentLabel + '\\RNN_Labels\\' 
        CommonFunctions.makeDirectory(data_file)
        CommonFunctions.makeDirectory(labels_file)
                        
        self.convertChunksIntoRNNFormat(userIndex , chunks_path , data_file , labels_file)
    
    def convertChunksIntoRNNFormat(self , userIndex, chunks_path, data_file1, labels_file1):
        """ """        
        files   = os.listdir(chunks_path)
        CommonFunctions.sort_nicely(files)
                
        data_file   = data_file1   + str(userIndex + 1) + '.csv'
        labels_file = labels_file1 + str(userIndex + 1) + '.csv'
        
        if os.path.exists(data_file):   #to avoid appending many time from different runs
            print ('data_file exists, so deleting ...')
            os.remove(data_file)
        else:
            print('creating RNN data file ...') 
            
        if os.path.exists(labels_file): #to avoid appending many time from different runs
            print ('labels_file exists, so deleting ...')
            os.remove(labels_file)
        else:
            print('creating RNN label file ...') 
                
        file_name = chunks_path + str(userIndex+1)+'.npy'
        AllChunks = np.load (file_name)
        
        for data_index in range (len(AllChunks)):    
                #if data_index < 3:
                
                chunk_data_labels = AllChunks[data_index]
                
                #save chunk data and chunks labels                    
                chunk_data_as_row, chunk_label = self.ReadChunkasVector(chunk_data_labels)
                
                #self.saveRNNDataLabels(chunk_data_as_row, chunk_label, data_file , labels_file)
                #print ('Saving data and labels of user ' , str(user_index + 1))
                self.saveRNNDataLabels2(chunk_data_as_row, chunk_label, data_file , labels_file)                
        
        #readRNNDataLabels(data_file , labels_file)
        """ """
                    
    def ReadChunkasVector(self, chunk_data_labels):
        """ """
        chunk_data_as_row = []   
        chunk_data        = chunk_data_labels[0]
        #print ('chunk_data  ' , np.array(chunk_data).shape)
        
        for chunk_index in range (len(chunk_data)):
                
                channel_data = chunk_data[chunk_index]
                for channel_index in range (len(channel_data)):
                    
                    frame_samples = channel_data[channel_index]
                    for sample_index in range (len(frame_samples)):
                        
                        sample = frame_samples[sample_index]
                        chunk_data_as_row.append(sample)
            
        chunk_label = chunk_data_labels[1]
    
        return chunk_data_as_row, chunk_label
    
    def saveRNNDataLabels(self, chunk_data_as_row, chunk_label , data_file , labels_file):
        """ """
                
        with open(data_file , 'a') as csvfile1:
            writer  = csv.writer(csvfile1)
            writer.writerow(chunk_data_as_row)
            csvfile1.close
         
        with open(labels_file , 'a') as csvfile2:
            writer   = csv.writer(csvfile2)
            writer.writerow(chunk_label)
            csvfile2.close()
    
    def saveRNNDataLabels2(self, chunk_data_as_row, chunk_label , data_file , labels_file):
        
        f = open(data_file,'a')
        f.write(str(chunk_data_as_row)) 
        f.write("\n") 
        f.close()
        
        f = open(labels_file,'a')
        f.write(str(chunk_label)) 
        f.write("\n") 
        f.close()

    def readRNNDataLabels(self, data_file , labels_file):
        """ """
        with open(data_file, 'r') as csvfile1:
            spamreader = csv.reader(csvfile1 , delimiter = ' ', quotechar = '|')
            for row in spamreader:
                print (', '.join(row))
    
        with open(labels_file, 'r') as csvfile2:
            spamreader = csv.reader(csvfile2 , delimiter = ' ', quotechar = '|')
            for row in spamreader:
                print (row)
                
if __name__ == "__main__":
    
    Preprocess_RNN  = Preprocess_RNN()
    
    userIndex = 0
    Preprocess_RNN.main_process(userIndex) 
    
    print ('Finished DEAP main process') 