import csv
import os
import scipy.io
import cv2
import decimal
import math
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from spectrum               import *
from sklearn.decomposition  import PCA
from scipy                  import signal
from sklearn                import decomposition

from Configurations_RNN     import Configurations
from Configurations_RNN     import Config_EEG
from Configurations_RNN     import Configurations
from Configurations_RNN     import Config_MFCC

Config      = Configurations()
Config_MFCC = Config_MFCC()
Config_EEG  = Config_EEG
Config      = Configurations()
""" *********************************************************************************************************************************** """
# define class of all related functions and parameters
class CommonFunctions_RNN(object):   

    def get_train_validate_partitions(self, AllData, train_index, validate_index):
        
        #print('TEST:', validate_index)
        train     = AllData[train_index] # Data with labels
        validate  = AllData[validate_index ] # Data with labels

        print('train : ' , train.shape)
        print('test  : ' , validate.shape)

        X_train = []
        Y_train = []
        for DataLable in train:
            X_train.append  ( DataLable[0] )    
            Y_train.append  ( DataLable[1] )
        
        #print ('X_train shape' , len( X_train))
        
        X_validate = []
        Y_validate = []
        for DataLable in validate:
            X_validate.append  ( DataLable[0] )    
            Y_validate.append  ( DataLable[1] )
        
        #print ('X_validate shape' , len( X_validate))

        return X_train, Y_train, X_validate, Y_validate
    
    def getChannelFrames(self, data):
        
        if Config.OverlapFlag:
            frames = self.do_framing_sigproc(data, Config_MFCC.samplerate)
        else:
            frames = self.do_framing(data, Config_MFCC.samplerate)
            
        return frames
     
    def getNumberOfFrames(self):
        """ 
        Get Number Of Frames in one channel
        """
        file_path      = Config_EEG.matlab_path + '\\s01.dat'
        data, lable    = self.ReadMatFile(file_path)
        
        if Config.OverlapFlag:
            frames = self.do_framing_sigproc(data[0][0], Config_MFCC.samplerate)
        else:
            frames = self.do_framing(data[0][0], Config_MFCC.samplerate)
            
        #frames         = self.do_framing_sigproc(data[0][0], Config_MFCC.samplerate)
        numberofframes = frames.shape[0]
        
        return numberofframes
    
    def getFrameSize(self):
        """ 
        Get Number Of Frames in one channel
        """
        file_path      = Config_EEG.matlab_path + '\\s01.dat'
        data, lable    = self.ReadMatFile(file_path)
        
        if Config.OverlapFlag:
            frames = self.do_framing_sigproc(data[0][0], Config_MFCC.samplerate)
        else:
            frames = self.do_framing(data[0][0], Config_MFCC.samplerate)
        
        #frames         = self.do_framing_sigproc(data[0][0], Config_MFCC.samplerate)
        
        framesize      = frames.shape[1]
        
        return framesize
    
    def getCurrentLablePath(self):
        
        current_lable   = Config.currentWorkingLabel
        
        if current_lable == 'valence':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.valence_onehot_labels_path #one hot!!!
        elif current_lable == 'arousal':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.arousal_onehot_labels_path #one hot!!!
        elif current_lable == 'liking':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.liking_onehot_labels_path #one hot!!!
        elif current_lable == 'dominance':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.dominance_onehot_labels_path #one hot!!!
        elif current_lable == 'MultiLabel':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.Valence_Arousal_MultiLabel #one hot!!!
        else:
            print ('************************ WRONG LABLE NAME, PLEASE CORRECT CURRENT WORKING LABLE *****************************')
            src_labels_path = ''         

        return src_labels_path
    
    def getCurrentLablePath_NotOneHot(self):
        
        current_lable   = Config.currentWorkingLabel
        if current_lable == 'valence':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.valence_labels_path #one hot!!!
        elif current_lable == 'arousal':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.arousal_labels_path #one hot!!!
        elif current_lable == 'liking':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.liking_labels_path #one hot!!!
        elif current_lable == 'dominance':
            src_labels_path = Config_EEG.labels_files_path + Config_EEG.dominance_labels_path #one hot!!!
        else:
            print ('************************ WRONG LABLE NAME, PLEASE CORRECT CURRENT WORKING LABLE *****************************')
            src_labels_path = ''         

        return src_labels_path
    
    """ ********************************************************************************************************************************** """         
    def ReadMatFile(self, file_path):
        """ 
        Read only one .dat file
        """
        import pickle
        pickle_in    = open(file_path, 'rb')
        example_dict = pickle.load(pickle_in, encoding='latin1')
        
        x = example_dict['data']
        y = example_dict['labels']
        
        #print ('data  shape : ' , x.shape)
        #print ('label shape : ' , y.shape)
        
        return x, y
    
    def ReadMatFileOriginal(self, file_path):
        """
        Read only one .mat file
        """
        mat_dict = {}
        mat_dict.update(scipy.io.loadmat(file_path))
        x = mat_dict['data']
        y = mat_dict['labels']
    
        return x, y
    
    def do_framing(self, data, Fs):
        """
        Given : data of one channel
        Return: frames of this data
        """               
        length      = len(data)
        framesize   = int(Config_MFCC.winlen * Fs)
        number_of_frames = int(length / framesize)
        
        # Declare a 2D matrix,with rows equal to the number of frames,and columns equal to the framesize or the length of each DTF
        frames = np.zeros((number_of_frames , framesize)) 
        count = 0
        for k in range( 0 , length, framesize):

                #if count == 0:
                if count < number_of_frames: #to have only 60 frames
                    if k + framesize < len(data): #not to go greater than last index of dat array
                        frames[count][:] = data[ k: k+framesize]
                        count += 1
        return frames
    
    def do_framing_sigproc(self,sig,Fs ):
        """Frame a signal into overlapping frames.
    
        :param sig: the audio signal to frame.
        :param frame_len: length of each frame measured in samples.
        :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        :returns: an array of frames. Size is NUMFRAMES by frame_len.
        """
        frame_len  = Config_MFCC.winlen  * Fs
        frame_step = Config_MFCC.winstep * Fs
        winfunc    = lambda x:np.ones((x,))
        #winfunc = np.hanning(frame_len)
        
        slen      = len(sig)
        frame_len = int(self.round_half_up(frame_len))
        frame_step= int(self.round_half_up(frame_step))
        
        if slen <= frame_len:
            numframes = 1
        else:
            numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))
    
        padlen    = int((numframes-1)*frame_step + frame_len)
        zeros     = np.zeros((padlen - slen,))
        padsignal = np.concatenate((sig,zeros))
    
        indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
        indices = np.array(indices,dtype=np.int32)
        frames  = padsignal[indices]
        win     = np.tile(winfunc(frame_len),(numframes,1))
        #win     = np.tile(winfunc,(numframes,1))
        return frames*win
    
    def round_half_up(self, number):
        return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
     
    def splitIntoTrainTest(self,train_test_file_path , CurrentClassLabel , split_ratio , split_random_state):
        
        # #Read full data
        train_test = CommonFunctions.Get_from_csv_Array(train_test_file_path)       
          
        #Read valence labels (one hot)
        labels = CommonFunctions.Get_from_csv_Array(CurrentClassLabel)       
        labels = labels [1:] #Deleting header :valence word        
                 
        #Split data and labels according to given ratio
        X_train, X_test, Y_train, Y_test = train_test_split(train_test, labels, test_size= split_ratio, random_state =split_random_state)          
    
        return X_train, X_test, Y_train, Y_test 
    
    def main_process(self):
        
        cwd         = os.path.dirname(os.path.realpath(__file__))
        file_name   = 's01.dat'
        file_path   = cwd + "//" + file_name
        
        print ('cwd       ' , cwd)
        print ('file_path ' , file_path)
        
        data, lable    = self.ReadMatFile( file_path)      #data of all channels and videos of one user
        print ('data.shape  ' , data.shape) 
        print ('label.shape ' , lable.shape) 
                        
if __name__ == "__main__":
     
    print ('Started ...')    
    CommonFunctions_RNN = CommonFunctions_RNN()
    CommonFunctions_RNN.main_process()
    print ('Finished ...')    

