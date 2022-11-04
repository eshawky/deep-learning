""" ************************************************************************************************************************************* """  
import tensorflow as tf
import os
from CommonFunctions import CommonFunctions
CommonFunctions      = CommonFunctions()

class Configurations(object):
	
	Main_Path		    = "C:\\0.PHD\\0.Phd Data\\0.DEAP Data\\Project_EEG_WorkingFile\\"
	DEAP_Working_Path   = Main_Path		    + "3DCNN\\3DCNN_channels_SD\\"  
	DEAP_Lables_path	= DEAP_Working_Path + "lables_per_frames.csv"

	num_frames_per_chunk   = 10
	num_of_users_per_batch = 1 
	floatPresicion         = 2
	
	currentWorkingLabel    = 'valence' #arousal valence MultiLabel
	hm_epochs              = 1000
	Kfold                  = 5
	
	OverlapFlag            = False
	
	ChannelNormalize       = False#False gives better results
	ApplyPCA               = False
	num_pca_Components     = 25
	num_lda_Components     = 300
	
	def __init__(self):
		""" """
		CommonFunctions.makeDirectory(self.DEAP_Working_Path)
		self.CurrentWorkingDirectory = os.getcwd()
        
class Config_Without (object):
	
	def __init__(self):
		
		Config = Configurations()
		if Config.currentWorkingLabel == 'valence':
			string = 'RevisedChannels'
		elif Config.currentWorkingLabel == 'arousal':
			string = 'RevisedChannels_Arousal'
		else:
			string = 'RevisedChannels_MultiLabel'
						
		self.DEAP_SD_Data_path   = Config.DEAP_Working_Path + string +'\\Subjects Batchs_' + str (Config.num_frames_per_chunk) + '_Revised\\'
		CommonFunctions.makeDirectory(self.DEAP_SD_Data_path)
		
		if Config.currentWorkingLabel == 'valence':
			string2 = 'Valence_' + str (Config.num_frames_per_chunk) + 'Chunks\\'
		elif Config.currentWorkingLabel == 'arousal':
			string2 = 'Arousal_' + str(Config.num_frames_per_chunk) + 'Chunks\\'
		else:
			string2 = 'Valence_Arousal_' + str(Config.num_frames_per_chunk) + 'Chunks\\'
		
		self.chunksPath  = Config.DEAP_Working_Path + string2
		#CommonFunctions.makeDirectory(self.chunksPath)

		self.FineTune         = False #True False
		self.FineTuneEpochs   = 2
		Number_Of_Noise_Files = 1 #should be 250		
		
class Config_With(object):
	
	def __init__(self):
		
		Config = Configurations()
		if Config.currentWorkingLabel == 'valence':
			string = 'SeparateNoisyChannels'
		elif Config.currentWorkingLabel == 'arousal':
			string = 'SeparateNoisyChannels_Arousal'
		else:
			string = 'SeparateNoisyChannels_MultiLabel'
							
		self.DEAP_SD_Data_path   = Config.DEAP_Working_Path + string +'\\Subjects Batchs_' + str (Config.num_frames_per_chunk) + '_Revised\\'
		#CommonFunctions.makeDirectory(self.DEAP_SD_Data_path)

	SNR					   = 5
	Number_Of_Noise_Files  = 50 #should be 250
	
class Config_Model_CNN_3D(object):
		
	Conv_strides    = 1
	Maxpool_Ksize   = 2
	Maxpool_Strides = 2

	Conv1_filterSize = 3
	Conv1_input      = 1    
	Conv1_output     = 8
	""" ********************** """
	Conv2_filterSize = 3
	Conv2_input      = Conv1_output    
	Conv2_output     = 16 
	""" ********************** """
	FC1_output       =  600 #1024
	FC2_output       =  200
	""" ********************** """
	WeightSeed            = 2342347823
	test_split_ratio      = 0.2
	split_random_state    = 42    #in split function
	PaddingType           = 'SAME'    
	learning_rate         = 0.003 
	""" ********************** """
	hm_epochs             = 5     #loss =1 548 611 435 at 10 6 727 621 420
	train_batch_size      = 100
	test_batch_size       = 100
	keep_rate_fc          = 0.5    # Dropout, probability to keep units
	keep_rate_maxpool     = 0.5
	""" ********************** """
	x                     = tf.placeholder('float32')#shape = [batch_size, frames_per_chunk, rows, columns, num_of_channels]
	y                     = tf.placeholder('float32')#shape = [batch_size, num_of_classes]
	y2                    = tf.placeholder('float32')#shape = [batch_size, num_of_classes]
	
	dropout_fc_holder     = tf.placeholder('float32')
	dropout_maxpool_holder= tf.placeholder('float32')
	lr                    = tf.placeholder('float')
	max_learning_rate     = 0.003
	min_learning_rate     = 0.0001
	decay_speed           = 2000.0
	normalize             = False #True or False   #for batch normalization
	zero_loss_ratio       = 3 #Flag for early stopping:: stop training when this flag reached (loss =0 for 3 times)
	std_epsilon           = 0.1
	#noisy_files_count     = 250

class Config_EEG(object):
    
    DEAP_Working_Path            = "C:\\0.PHD\\0.Phd Data\\0.DEAP Data\\Project_DEAP_WorkingFile\\"
    matlab_path                  = DEAP_Working_Path + 'data_preprocessed_python'
    video_data                   = DEAP_Working_Path + 'video_data_files_path\\'
    labels_files_path            = DEAP_Working_Path + 'labels_files_path\\'
    train_test_file_path         = DEAP_Working_Path + 'train_test_DEAP.csv'
    
    Twente                       = DEAP_Working_Path + 'Twente.csv'
    Geneva                       = DEAP_Working_Path + 'Geneva.csv'
    Twente_GenevaFlag            = 'No' #Twente / Geneva / No
    convertPeripheralChannels    = False #ALWAYES FALSE
        
    data_csv_path                = 'channel_data_1_1.csv'                  
    labels_csv_path              = 'labels.csv'                            
    updated_labels_csv_path      = 'labels_in_ranges.csv'                  
    valence_labels_path          = 'valence_labels.csv'                    
    arousal_labels_path          = 'arousal_labels.csv'                  
    liking_labels_path           = 'liking_labels.csv'                    
    dominance_labels_path        = 'dominance_labels.csv'                  
    valence_onehot_labels_path   = 'valence_onehot_labels.csv'             
    arousal_onehot_labels_path   = 'arousal_onehot_labels.csv'            
    liking_onehot_labels_path    = 'liking_onehot_labels.csv'             
    dominance_onehot_labels_path = 'dominance_onehot_labels.csv'        
    Valence_Arousal_MultiLabel   = 'Valence_Arousal_MultiLabel.csv'

    NumOfClasses          = 2    #(2 for low and high valence and 4 for low/high valence/arousal)  
    SizeOfDataPerChannel  = 8064
    NumOfChannels         = 32
    NumOfVideos           = 40
    NumOfUsers            = 32
    NumOfSamples          = 1280   
    
    roundValue            = 5 
    csv_files_extention   = '.csv'
    
    userId      = "userID"
    videoId     = "videoID"
    channelId   = "channelID"
    data        = "data"
    label       = "label"
    valenceId   = 'valence'
    arousalId   = 'arousal'
    likingId    = 'liking'
    dominanceId = 'dominance'
    class1      = 'low'
    class2      = 'high'
    class3      = 'medium'
  
class Config_AutoEncoder_mnist(object):
	""" """
	WeightSeed   = 2342347823
	num_input    = 784 # MNIST data input (img shape: 28*28) 
	num_hidden_1 = 256 # 1st layer num features
	num_hidden_2 = 128 # 2nd layer num features (the latent dim)
	X            = tf.placeholder("float32", [None, num_input]) #tf.placeholder("float", [None, num_input])    
	
	learning_rate= 0.01
	num_steps    = 10
	batch_size   = 200
	display_step = 2
	n_classes    = 10
	n_train      = 60000
	n_test       = 10000
	  
class Config_AE_DEAP(object):
	""" """
	WeightSeed   = 2342347823
	num_input    = 5360        # MNIST data input (img shape: 40*134) 
	num_hidden_1 = 2048        # 256 1st layer num features
	num_hidden_2 = 1024        # 128 2nd layer num features (the latent dim)
	X            = tf.placeholder("float32", [None, num_input]) #tf.placeholder("float", [None, num_input])    
	Y            = tf.placeholder("float32") #tf.placeholder("float", [None, num_input])    
	learning_rate= 0.01
	batch_size   = 200
	#n_classes    = 2
	n_train      = 2000
	n_test       = 400
	display_step = 1
	hm_epochs    = 10

	def __init__(self):
		
		Config     = Configurations()
		config_C3D = Config_Model_CNN_3D ()
		 
		self.AE_Data_Path = Config.DEAP_Working_Path + 'AE_Data_Vector\\'
		CommonFunctions.makeDirectory(self.AE_Data_Path)
		
		self.currentConfirgurations = self.AE_Data_Path + Config.currentWorkingLabel + '\\' + str (Config.num_frames_per_chunk) +'Chunks_' + str(self.hm_epochs) + 'AE_Epochs_' + str(self.num_hidden_2) +'AE_Features_3DCNN_' + str(Config.hm_epochs) + '_Epochs_' + str(config_C3D.FC1_output) + 'Features' 
		CommonFunctions.makeDirectory(self.currentConfirgurations)
		
		str0 = '\\out_models\\'
		self.AE_Model_Path = self.currentConfirgurations + str0
		CommonFunctions.makeDirectory(self.AE_Model_Path)
		
		#create path for raw frames of EEG signals: mat of 40*134=5360 samples
		str1             = '\\Raw_Frames\\'
		self.Raw_Frames  = self.AE_Data_Path +  Config.currentWorkingLabel + str1
		CommonFunctions.makeDirectory(self.Raw_Frames)
		
		#create path for AE features
		str2   = '\\Features\\'
		self.AE_Features   = self.currentConfirgurations +  str2
		CommonFunctions.makeDirectory(self.AE_Features)
		
		#create path for AE chunks
		str3           = '\\Chunks\\'
		self.AE_Chunks = self.currentConfirgurations + str3  
		CommonFunctions.makeDirectory(self.AE_Chunks)		
		
		str4 = '\\accuracy_file\\' 
		self.AE_AccuracyFile = self.currentConfirgurations + str4  
		CommonFunctions.makeDirectory(self.AE_AccuracyFile)	
		
class Config_AE_DEAP_Matrix (object):
	""" 
	Get 50 AE features for each frame and 
	Then combine them for all 40 channels to create matrix 40*50
	"""
	WeightSeed   = 2342347823
	num_input    = 134        # MNIST data input (img shape: 40*134) 
	num_hidden_1 = 100        # 100 1st layer num features
	num_hidden_2 = 50         # 50  2nd layer num features (the latent dim)
	X            = tf.placeholder("float32", [None, num_input]) #tf.placeholder("float", [None, num_input])    
	Y            = tf.placeholder("float32") #tf.placeholder("float", [None, num_input])    
	learning_rate= 0.01
	batch_size   = 200
	display_step = 1
	hm_epochs    = 50

	def __init__(self):
		
		Config     = Configurations()
		config_C3D = Config_Model_CNN_3D ()

		self.AE_Data_Path = Config.DEAP_Working_Path + 'AE_Data_Matrix\\'
		
		self.currentConfirgurations = self.AE_Data_Path + Config.currentWorkingLabel + '\\' + str (Config.num_frames_per_chunk) +'Chunks_' + str(self.hm_epochs) + 'AE_Epochs_' + str(self.num_hidden_2) +'AE_Features_3DCNN_' + str(Config.hm_epochs) + '_Epochs_' + str(config_C3D.FC1_output) + 'Features' 
				
		str0 = '\\AE_Models\\'
		self.AE_Model_Path = self.currentConfirgurations + str0
		
		#create path for raw frames of EEG signals: mat of 40*134=5360 samples
		str1             = '\\Raw_Frames\\'
		self.Raw_Frames  = self.AE_Data_Path +  Config.currentWorkingLabel + str1
		
		#create path for AE features
		str2   = '\\AE_Features\\'
		self.AE_Features   = self.currentConfirgurations +  str2
		
		#create path for AE chunks
		str3           = '\\AE_Chunks\\'
		self.AE_Chunks = self.currentConfirgurations + str3  
		  		
		""" """
		self.ApplyAE = False  #True False
		if self.ApplyAE:
			str4 = '\\With_AE_accuracy_file\\' 
			self.AE_AccuracyFile = self.currentConfirgurations + str4
			self.TrainTestSets   = self.currentConfirgurations + "\\AE_TrainTestSets\\"
			self.modelPath       = self.currentConfirgurations + "\\AE_Training_Models"
		else:
		    str4 = '\\Without_AE_accuracy_file\\' 
		    self.AE_AccuracyFile = self.currentConfirgurations + str4
		    self.TrainTestSets   = self.currentConfirgurations + "\\TrainTestSets\\"
		    self.modelPath       = self.currentConfirgurations + "\\Training_Models"

		self.TrainDataPath   = self.TrainTestSets + "Train\\"
		self.TestDataPath    = self.TrainTestSets + "Test\\"
		self.Log_Dir         = "/tmp/deap_demo/SD_Users/"  
		
		CommonFunctions.makeDirectory(self.AE_AccuracyFile)	
		CommonFunctions.makeDirectory(self.AE_Chunks)		
		CommonFunctions.makeDirectory(self.AE_Features)
		CommonFunctions.makeDirectory(self.Raw_Frames)
		CommonFunctions.makeDirectory(self.currentConfirgurations)
		CommonFunctions.makeDirectory(self.AE_Model_Path)
		CommonFunctions.makeDirectory(self.AE_Data_Path)

class Config_DFT(object):
	""" """
	def __init__(self):
		
		self.DeepModel        = 'C3D' #C3D / RNN
		
		self.FreqType         = 'NO'    # mfcc/fft/dft/No
		self.CalcDerivative   = True
		
		self.mfccRange        = 'frame' # frame/signal
		
		self.ExtractFreqBands = False
		self.VisualizeBands   = False
		
		self.ApplyDFTAfter  = True
		self.saveDFT        = False
		
		self.ApplyDFTBefore = False	
		self.saveC3D        = True				
		
		self.DFTComplex     = True
		self.svm_linear     = True
		self.DFT_Classifier = 'svm' #svm or softmax
		self.kernel         = 'rbf'
		self.svm_classifier_type = 'General' #OneVsRest or OneVsOne or General
				
		Config              = Configurations()
		config_C3D          = Config_Model_CNN_3D ()
		self.DFT_Data_Path  = Config.DEAP_Working_Path + 'DFT_Results\\'
		
		self.currentConfirgurations   = self.DFT_Data_Path + Config.currentWorkingLabel + '\\'  +'DFT_Features_3DCNN_' + str(Config.hm_epochs) + '_Epochs_' + str(config_C3D.FC1_output) + 'Features' 
		self.Without_DFT_AccuracyFile = self.currentConfirgurations + "\\Without_DFT_accuracy_file\\"  				
		CommonFunctions.makeDirectory(self.currentConfirgurations)
		CommonFunctions.makeDirectory(self.Without_DFT_AccuracyFile)
		
		if self.ApplyDFTBefore:
			self.DFTB4_Chunks_Path     = self.currentConfirgurations + "\\DFTB4_Chunks_Path\\"
			CommonFunctions.makeDirectory(self.DFTB4_Chunks_Path)
			
		if self.saveDFT :  
			self.train_dft_path           = self.currentConfirgurations + "\\train_dft_features\\"
			self.test_dft_path            = self.currentConfirgurations + "\\test_dft_features\\"
			CommonFunctions.makeDirectory(self.train_dft_path)
			CommonFunctions.makeDirectory(self.test_dft_path)
		
		if self.saveC3D : 
			self.train_c3d_path           = self.currentConfirgurations + "\\train_c3d_features\\"
			self.test_c3d_path            = self.currentConfirgurations + "\\test_c3d_features\\"
			CommonFunctions.makeDirectory(self.train_c3d_path)
			CommonFunctions.makeDirectory(self.test_c3d_path)
			
		if self.DFT_Classifier == 'svm':
		
			if self.svm_linear:	
				self.currentConfirgurations + self.currentConfirgurations + "_Linear"
				
				if self.svm_classifier_type == 'OneVsOne':
					self.DFT_AccuracyFile         = self.currentConfirgurations + "\\DFT_SVM_Linear_OneVsOne_accuracy_file" 
				elif self.svm_classifier_type == 'OneVsRest':
					self.DFT_AccuracyFile         = self.currentConfirgurations + "\\DFT_SVM_Linear_OneVsRest_accuracy_file"
				else:
					self.DFT_AccuracyFile         = self.currentConfirgurations + "\\DFT_SVM_Linear_General_accuracy_file"	
			else:
				self.currentConfirgurations + self.currentConfirgurations + "_NonLinear"
				self.DFT_AccuracyFile         = self.currentConfirgurations + "\\DFT_SVM_NonLinear_accuracy_file"  
				
			if self.DFTComplex:
				self.DFT_AccuracyFile         = self.DFT_AccuracyFile + "_Complex\\"  
			else:
				self.DFT_AccuracyFile         = self.DFT_AccuracyFile + "_Real\\"  

			CommonFunctions.makeDirectory(self.DFT_AccuracyFile)
		
		else:
			self.DFT_SoftMax_Accuracy     = self.currentConfirgurations + "\\DFT_SoftMax_accuracy_file\\"
			CommonFunctions.makeDirectory(self.DFT_SoftMax_Accuracy)
		
		if self.FreqType == 'mfcc':
			self.MFCC_Raw_Frames = self.currentConfirgurations + "\\MFCC_Raw_Frames\\"                                   
			CommonFunctions.makeDirectory(self.MFCC_Raw_Frames)
				
class Config_MFCC(object):
	""" """			
	samplerate   = 128    # len(frame), or len(signal) : samppling frequency of DEAP EEG data
	winlen       = 1.0    # 1.00 * samplingRate = 128 samples per frame
	winstep      = winlen/2   # To get a sample index in frame 1 which is a starting sample of frame 2: 0.5  * samplingRate = 64 samples for overlap:nframes = 125(63+63): to decrease nframes, decrease overlap
	
	numcep       = 13     #13,26,39
	nfilt        = 40     #26,52,78     
	nfft         = 512    #512 should be large
	
	lowfreq      = 4      #0
	highfreq     = 45     #None
	preemph      = 0.98   #0.97
	ceplifter    = 22     #22
	appendEnergy = True   #True
	
	fft_points   = 512
	ownFraming   = False  #has no overlap
	
class Config_RNN(object):
	""" """
	Config             = Configurations()
	currentWorkingLabel= Config.currentWorkingLabel   
	Config_EEG         = Config_EEG()
	Config_Without     = Config_Without()
	chunks_path        = Config_Without.DEAP_SD_Data_path     
	
	rnn_path           = 'C:\\0.PHD\\0.Phd Data\\0.DEAP Data\\Project_RNN_WorkingFile\\'
	currentLabelPath   = rnn_path + currentWorkingLabel + "\\"
	data_file          = currentLabelPath + 'RNN_Chunks\\' 
	labels_file        = currentLabelPath + 'RNN_Labels\\' 
	
	RNN_Results        = currentLabelPath + 'RNN_Results\\'
	rnn_accuracy_file  = RNN_Results      + 'Accuracy_File\\'
	
	CommonFunctions.makeDirectory(data_file)
	CommonFunctions.makeDirectory(labels_file)
	CommonFunctions.makeDirectory(RNN_Results)
	CommonFunctions.makeDirectory(rnn_accuracy_file)
	
	chunk_size         = 32*128*1    #fsize*nchannels*windowsize: 4096=128*32, 5120=128*40, 5360=134*40
	n_chunks           = Config.num_frames_per_chunk  
	batch_size         = 100          #int  ( n_chunks / 2 )  #Config.RNN.batch_size 
	
	num_layers         = 2
	num_hidden_nodes1  = 64
	num_hidden_nodes2  = 32
	
	input_dropout      = 0.2
	output_dropout	   = 0.2
	
	hm_epochs          = 35
	nclasses		   = Config_EEG.NumOfClasses 

	WeightSeed         = 2342347823
	split_ratio        = 0.2
	split_random_state = 42   #in split function
	zero_loss_ratio    = 3
    
	x                  = tf.placeholder('float32', [None , n_chunks ,chunk_size])
	y                  = tf.placeholder('float32')
	
class Config_C3D_RNN(object):
	
	Config             = Configurations()
	currentWorkingLabel= Config.currentWorkingLabel 
	
	rnn_path           = 'C:\\0.PHD\\0.Phd Data\\0.DEAP Data\\Project_EEG_WorkingFile\\3DCNN\\3DCNN_channels_SD\DFT_Results\\Feature_Fusion\\'
	currentLabelPath   = rnn_path + currentWorkingLabel + "\\"
	
	C3D_RNN_Results       = currentLabelPath + 'C3D_RNN_Results\\'
	c3d_rnn_accuracy_file = C3D_RNN_Results  + 'Accuracy_File\\'
	
	CommonFunctions.makeDirectory(C3D_RNN_Results)
	CommonFunctions.makeDirectory(c3d_rnn_accuracy_file)
	
	chunk_size         = 1800    #fsize*nchannels*windowsize: 4096=128*32, 5120=128*40, 5360=134*40
	n_chunks           = 1  
	
	x                  = tf.placeholder('float32', [None ,  n_chunks, chunk_size])
	y                  = tf.placeholder('float32')
	
	n_input            = chunk_size
	num_hidden_nodes1  = 1000
	num_hidden_nodes2  = 500
	
	Config_EEG         = Config_EEG()
	nclasses           = Config_EEG.NumOfClasses  
	hm_epochs          = 50
