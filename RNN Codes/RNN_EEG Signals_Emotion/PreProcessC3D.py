from datetime import datetime
now = datetime.now()
""" ************************************************************************************************************************************* """  
import sys
sys.path.append("../")

sys.path.append("../PreprocessSignal/")
from Preprocessing_Net  import Preprocessing_Net
from Configurations_All import Config_MFCC
Preprocessing_Net = Preprocessing_Net()
Config_MFCC       = Config_MFCC()

import csv
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split 
import cv2
import tensorflow as tf
import random
import math
import sklearn as sk
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import KFold
""" ************************************************************************************************************************************* """  
from CommonFunctions          import CommonFunctions
from CommonFunctions_RNN      import CommonFunctions_RNN
from Configurations_RNN       import Configurations
from Configurations_RNN       import Config_Model_CNN_3D
from Configurations_RNN       import Config_EEG 
from Configurations_RNN       import Config_With
from Configurations_RNN       import Config_Without
""" ************************************************************************************************************************************* """  
Config            = Configurations()
config_C3D        = Config_Model_CNN_3D()
Config_Without    = Config_Without()
Config_EEG        = Config_EEG
CommonFunctions   = CommonFunctions()
CommonFunctions_RNN=CommonFunctions_RNN()
""" ************************************************************************************************************************************* """  
class PreProcessC3D(object):
    
    def getAllDataPerUser(self, batch , user_index):
        
        AllDataPerBatch = []
        n_chunks        = Config.num_frames_per_chunk
                 
        """ Perpare User Data"""
        user_data,  numberofframes      = self.getDataPerUser(batch) #(2400, 40, 134)because number of videos per user = 40, number of frames per video = 60 so 2400        
        print ('Total frames in one user ', np.array (user_data).shape)#videos*frames=16800 frames(5users*40videos*84frames)
      
        print('number of videos           ' , Config_EEG.NumOfVideos)
        print('number of frames per video ' , numberofframes, '.. should be 63 or 125, but I delete 1st 3sec .. ')
        print ('Total frames in one user  ' , Config_EEG.NumOfVideos*numberofframes)
       
        """ Perpare User Lables """
        currentLablePath = CommonFunctions_RNN.getCurrentLablePath()
        
        lables_per_user  = self.getLablesPerUser (currentLablePath , user_index)        
        print ('Lables per user', lables_per_user.shape )#users in batch*videos*frames=16800 frames(5users*40videos*84frames)
        
        lables_per_frames = self.getLablesPerFrames(lables_per_user, numberofframes)
        print ('Lables per frames', lables_per_frames.shape )#users in batch*videos*frames=16800 frames(5users*40videos*84frames)
 
        lable_index = 0;
        user_data = np.array(user_data)
            
        for chunk_index in range(0, user_data.shape[0], n_chunks):#n_chunks*4,n_chunks ):#
              
            #get chunk data   
            chunk_data  = user_data[chunk_index : chunk_index + n_chunks] 
                           
            if chunk_data.shape[0] == n_chunks:
                #print ('Current chunk index ', chunk_index)
     
                #get chunk label
                chunk_lables = lables_per_frames[chunk_index : chunk_index + n_chunks] 
                
                currentLabel = Config.currentWorkingLabel
                if currentLabel =='MultiLabel':  
                    chunk_lable_oneHot = self.getLablePerChunk_MultiLabels(chunk_lables)              
                else:#if currentLabel =='valence' or currentLabel =='arousal' :
                    chunk_lable_oneHot = self.getLablePerChunk(chunk_lables)
                #print ('chunk_lable_oneHot')
                #print (chunk_lable_oneHot)
                chunk_lable_oneHot = np.array(chunk_lable_oneHot)
                
                #append videoFrames of one chunk and its one hot lable in AllData List
                AllDataPerBatch.append([ chunk_data , chunk_lable_oneHot ])
             
            #else pad last frame with zeros. how?? not important: its ok to neglect last 24 samples..
               
        print ('AllDataPerBatch.shape             '  , len(AllDataPerBatch)             )
        print ('AllDataPerBatch.shape[0]          '  , len(AllDataPerBatch[0])          )
        print ('AllDataPerBatch.shape[0][0]       '  , len(AllDataPerBatch[0][0])       )        
        print ('AllDataPerBatch.shape[0][0][0]    '  , len(AllDataPerBatch[0][0][0])    )
        print ('AllDataPerBatch.shape[0][0][0]    '  , len(AllDataPerBatch[0][0][0][0]) )
     
        return AllDataPerBatch
    
    def getDataPerUser(self, batch):
        
        """ This function for batch with one or many users """
        eeg_path             = Config_EEG.matlab_path
        users                = os.listdir(eeg_path)
        CommonFunctions.sort_nicely(users)
        user_frames         = []
         
        for userId , user in enumerate (users):#32 users
             
            if user in batch:#work on users in current batch only
                
                file_path    = eeg_path + '\\' + user
                data, lable  = CommonFunctions_RNN.ReadMatFile(file_path)      #data of all channels and videos of one user

                for videoId in range(Config_EEG.NumOfVideos):#40 videos #work on one video for now                    
                    
                    """ Apply Pre-processing for original data for one video """
                    video_data = data[videoId]
                    video_data = Preprocessing_Net.preprocess(video_data , Config_MFCC.samplerate)
                    #video_data[0:32] = Preprocessing_Net.preprocess(video_data[0:32] , Config_MFCC.samplerate)
                    #print ('data after ' , np.array( video_data ).shape)
                      
                    video_frames   = []
                    channelIndeces = np.arange(0 , Config_EEG.NumOfChannels)

                    #channelIndeces = np.arange(32 , 32 + Config_EEG.NumOfChannels) #Eye: nchannels=4
                    #channelIndeces = np.arange(35 , 35 + Config_EEG.NumOfChannels) #EMG: nchannels=4

                    if Config_EEG.Twente_GenevaFlag == 'Twente':
                        channelIndeces = self.sortChannelsTwente()
                    
                    if Config_EEG.Twente_GenevaFlag == 'Geneva':
                        channelIndeces = self.sortChannelsGeneva()
                    
                    """ If you need specific channels (EEG+Eye or EEG+GSR or Eye+GSR or one set only ) """
                    #arr1 = np.arange(0, Config_EEG.NumOfChannels-4)
                    #arr2 = np.arange(Config_EEG.NumOfChannels, Config_EEG.NumOfChannels+4)
                    #arr  = np.concatenate((arr1,arr2))
                    #print ('channel indeces : ' , arr)
                    
                    #for channelId in range(Config_EEG.NumOfChannels):#40 channel
                    #for channelId in range(32, 32 + Config_EEG.NumOfChannels):#eye only if nchannels=4
                    #for channelId in arr :
                            
                    for channelId in channelIndeces:#40 channel
                        
                        dataVector = video_data[channelId]
                        
                        """ removing first 3 seconds which are not emotions """
                        dataVector = dataVector[384:]
                        #print ('length of data ', len(dataVector))
                        
                        frames = CommonFunctions_RNN.getChannelFrames(dataVector)
                        #print ('length of frames ', np.array(frames).shape)
                                                
                        video_frames.append(frames)                 
                                  
                    video_frames = np.array(video_frames)
                                         
                    #update to my format
                    for frameIndex in range (video_frames[0].__len__()):#get number of frames per channel
                         
                        new_video_frames = []
                        for channelId in range (video_frames.__len__()):
                            currentFrames = video_frames[channelId][frameIndex]
                            
                            #""" Normalize frame from all channels """
                            #mean          = np.mean(currentFrames , axis=0) #axis=0 get mean of a sample from all channels
                            #currentFrames = currentFrames - mean
                                
                            new_video_frames.append(currentFrames)
                        
                        if Config_EEG.convertPeripheralChannels:
                            new_video_frames = self.convertPeripheralChannels(new_video_frames)
                                                                         
                        user_frames.append(new_video_frames)
              
        return user_frames, len(frames)
    
    def getLablesPerUser(self, currentLablePath, user_index):
        """ Read labels_per_frames from file  """
       
        data_csv         = CommonFunctions.Get_from_csv_DataFrame(currentLablePath)                        
        #print ('******************************************************************** data csv ' , data_csv)
        
        step             =  Config_EEG.NumOfVideos  #5users per batch*40 videos each*84 number of frames in one video
        start            = (user_index-1) * step 
        end              = start + step
        
        lables_per_user = data_csv.iloc[start : end]  #because last batch is longer than other batches 7*40*84: 7 users not 5
        lables_per_user = np.array (lables_per_user)
        
        #print ('lables_per_userrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        #print ( lables_per_user)
        
        return lables_per_user
    
    def getLablesPerFrames(self, lables_per_user, numberofframes):
                
        #get lables per frames
        lables_per_frames = []
        for i in range (lables_per_user.shape[0]):
            for j in range (numberofframes):
                lables_per_frames.append( lables_per_user[i] )
           
        lables_per_frames = np.array (lables_per_frames)   
        #print (lables_per_frames[60*3:60*4])#read from lables_per_frames file
        
        #print ('lables_per_frames' , lables_per_frames)
        return lables_per_frames           
    
    def getLablePerChunk(self, lables_per_one_chunk):        
         
        n_chunks = Config.num_frames_per_chunk 
        count    = 0
                
        for lable in lables_per_one_chunk:
            
            if lable[0] == 0: #means class 2 because label should be [0 1] means class 2 and I ask about first part
                count += 1
                
        if count > int (n_chunks/2) :
            currentLable = 1
        else:
            currentLable = 0    
        
        #convert lable to one hot
        if currentLable == 1:
            chunk_lable_oneHot = [0, 1] 
        else:
            chunk_lable_oneHot = [1, 0]
        
        #print (chunk_lable_oneHot)    
        return chunk_lable_oneHot

    def getLablePerChunk_MultiLabels (self, lables_per_one_chunk):        
         
        n_chunks = Config.num_frames_per_chunk 
        ValenceCount = 0
        ArousalCount = 0
        
        #Valence        
        for lable in lables_per_one_chunk:            
            if lable[0] == 0: #means class 2 because label should be [0 1] means class 2 and I ask about first part
                ValenceCount += 1
            
            if lable[2] == 0: #means class 2 because label should be [0 1] means class 2 and I ask about first part
                ArousalCount += 1
                    
        if ValenceCount > int (n_chunks/2) :
            currentLable = 1
        else:
            currentLable = 0            
        if currentLable == 1:
            chunk_valence_lable_oneHot = [0, 1] 
        else:
            chunk_valence_lable_oneHot = [1, 0]

     
        if ArousalCount > int (n_chunks/2) :
            currentLable = 1
        else:
            currentLable = 0            
        if currentLable == 1:
            chunk_arousal_lable_oneHot = [0, 1] 
        else:
            chunk_arousal_lable_oneHot = [1, 0]    

        chunk_lable_oneHot = []
        chunk_lable_oneHot.append(chunk_valence_lable_oneHot[0])
        chunk_lable_oneHot.append(chunk_valence_lable_oneHot[1])
        chunk_lable_oneHot.append(chunk_arousal_lable_oneHot[0])
        chunk_lable_oneHot.append(chunk_arousal_lable_oneHot[1])
                
        return chunk_lable_oneHot
    
    def convertPeripheralChannels(self, FrameFromAllChannels):
        """ """
        list = []
        for i in range (len (FrameFromAllChannels)):
            if i>=32:
                FrameFromAllChannels[i] = math.pow(10, 9)/FrameFromAllChannels[i]
                list.append(FrameFromAllChannels[i])
            else:
                list.append(FrameFromAllChannels[i])
                
        return list        
    
    def sortChannelsTwente(self):
        channelIndeces = [1  ,2 , 4 , 3 , 6 , 5 , 8 , 7 , 10, 9,
                          12, 11, 16, 13, 14, 15, 32, 31, 29, 30,
                          27, 28, 25, 26, 22, 23, 20, 21, 18, 17,
                          19, 24,
                          32, 33, 34, 35, 36, 37, 38, 39]
        
        return channelIndeces
        
    def sortChannelsGeneva(self):
        channelIndeces = [1,  2,  4 , 3 , 6 , 5,  8 , 7 , 10, 9, 
                          12, 11, 14, 15, 16, 13, 30, 29, 31, 27,
                          28, 25, 26, 32, 23, 24, 21, 22, 19, 20, 
                          18, 17,
                          32, 33, 34, 35, 36, 37, 38, 39]
    
        return channelIndeces
    
    def main_process(self, userIndex):
        """ 1-prepare all frames in one list/array: takes 10 minutes"""
        eeg_path = Config_EEG.matlab_path
        users    = os.listdir(eeg_path)
        CommonFunctions.sort_nicely(users)
                    
        """ 3-prepare full Data for 3D CNN THEN save them in .npy file format""" 
        
        #for userIndex in range (Config_EEG.NumOfUsers):
            
        #if i == 0: 
        batch      = users[ userIndex : userIndex + 1] #contains names of user files like s01.mat
        user_index = userIndex + 1
        print ('I.m in user index ***************************  ' , user_index )
        Batch_AllData = self.getAllDataPerUser ( batch, user_index)
        np.save (Config_Without.DEAP_SD_Data_path + str(userIndex+1)+'.npy' , Batch_AllData)#saving takes about 3 minutes
                
if __name__ == "__main__":
    
    PreProcessData  = PreProcessData()
    
    userIndex = 0
    #PreProcessData.main_process(userIndex) #1 for kfold and 2 for random split
    
    input = np.array([[-5, 10,2], 
                      [-2, -3,3], 
                      [-4, -9,1], 
                      [7 , 11,-3], 
                      [12, 6,-1], 
                      [13, 4,5]])
    
    mean = np.mean(input, axis=0)
    print ('mean is : ' , mean)
    #data = input - np.mean(input, axis=0)
    