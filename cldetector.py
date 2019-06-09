import numpy as np
import tensorflow as tf

class CloudDetector():
    'Predict clouds'
    def __init__(self, input, modelType='pixel'):
        'Initialization'
        self.errorFlag = False
        self.n_class = 3
        self.n = 5
        self.channels = 14
        self.segment_dim = (self.n, self.n, self.channels)
        
        self.modelType = modelType
        self.checkModelType()
        
        self.input = input       
        self.spatial_x = self.input.shape[0]
        self.spatial_y = self.input.shape[1]
        self.checkInputDim()
        
        if self.errorFlag:
            print('Initialization stopped due to incorrect input shape or values.')
        else:
            self.preprocessInput()        
            self.loadModel()
    
    def checkModelType(self):
        'Sanity check for modelType variable' 
        if not(self.modelType == 'pixel' or self.modelType == 'cnn'):
            self.errorFlag = True
            print('Not implemented model type.')
 
    def checkInputDim(self):
        'Sanity check for input variable'
        if self.input.ndim!=3:
            self.errorFlag = True
            print('Incorrect input shape.')
        if self.modelType == 'pixel':
            if not(self.spatial_x>0 and self.spatial_y>0 and self.input.shape[2] == self.channels):
                self.errorFlag = True
                print('Incorrect input dimension for model type.')
        if self.modelType == 'cnn':
            if not(self.spatial_x>=self.n and self.spatial_y>=self.n and self.input.shape[2] == self.channels):
                self.errorFlag = True
                print('Incorrect input dimension for model type.')
            
    def preprocessInput(self):
        'Process input image for prediction step'
        if self.modelType == 'pixel':
            'Normalization'
            self.x = (self.input / 2.0) - 0.5
            
            '2D image -> 1D series of pixels '
            self.x = np.reshape(self.x, (self.spatial_x * self.spatial_y, self.channels))
        if self.modelType == 'cnn':
            'Normalization'
            self.x = (self.input / 2.0) - 0.5
            
            'Set up ID system for generator'
            sample_number_per_row = self.spatial_x - self.n + 1
            sample_number_per_column = self.spatial_y - self.n + 1
            sample_per_image = sample_number_per_row * sample_number_per_column
            ID_list = np.arange(sample_per_image)
            
            'Init generator'            
            params = {'dim': self.segment_dim,
                      'batch_size': 32768,
                      'n_classes': self.n_class}
            self.predict_generator = DataGenerator(self.x, ID_list, **params)

    def loadModel(self):
        if self.modelType == 'pixel':  
            self.model = tf.keras.models.load_model('pixelwise_v8.h5')
            self.model.load_weights('pixelwise_v8_weights.hdf5')
        if self.modelType == 'cnn':
            print('loadModel() - cnn branch: not implemented')  
            self.model = tf.keras.models.load_model('cnn_v0.h5')
            self.model.load_weights('cnn_weights_v0.hdf5') 

    def predict(self):
        'Return prediction'
        if self.errorFlag:
            print('No prediction due to error.')
        else:
            if self.modelType == 'pixel':  
                y = self.model.predict(self.x)
                return np.reshape(y, (self.spatial_x, self.spatial_y, self.n_class))
            if self.modelType() == 'cnn':
                print('predict() - cnn branch: not implemented')
                return self.model.predict_generator(self.predict_generator)

    def updateInput(self, input):
        'Return prediction'  
        self.input = input     

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input, list_IDs, batch_size=32, dim=(32,32,32), n_classes=10):
        'Initialization'
        self.input = input
        self.spatial_x = self.input.shape[0]
        self.spatial_y = self.input.shape[1]
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Load batch data
        half_seg = int(self.dim[0] / 2)
        sample_number_per_row = self.spatial_x - self.dim[0] + 1
        sample_number_per_column = self.spatial_y - self.dim[1] + 1
        sample_per_image = sample_number_per_row * sample_number_per_column
        for i, ID in enumerate(list_IDs_temp):
            # # Cropping out the segment from input image based on ID
            x = (ID % sample_per_image) % sample_number_per_row + half_seg
            y = math.floor(ID / sample_number_per_row) + half_seg
            X[i,] = self.input[x-half_seg:x+half_seg+1, y-half_seg:y+half_seg+1, :]

        return X, y