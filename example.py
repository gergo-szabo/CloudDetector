import numpy as np
import cldetector as cld
import matplotlib.pyplot as plt

if __name__ == '__main__':
    'Loading data'
    test_set = np.load('test/test_set.npz')
    input = test_set['arr_0']
    
    'Predicting'
    glassball1 = cld.CloudDetector(input, modelType='pixel')
    output1 = glassball1.predict()
    
    glassball2 = cld.CloudDetector(input, modelType='cnn')
    output2 = glassball2.predict()
    
    'Show results'
    plt.figure(figsize=(12,5))
    plt.suptitle('Prediction of pixel information based model', fontsize=16)
    
    plt.subplot(1, 3, 1)
    plt.imshow(output1[:, :, 0])
    plt.title('Probability map of land (and water)')

    plt.subplot(1, 3, 2)
    plt.imshow(output1[:, :, 1])
    plt.title('Probability map of cloud')
    
    plt.subplot(1, 3, 3)
    plt.imshow(output1[:, :, 2])
    plt.title('Probability map of shadows')

    plt.show()
    
    plt.figure(figsize=(12,5))
    plt.suptitle('Prediction of area information based model(cnn)', fontsize=16)
    
    plt.subplot(1, 3, 1)
    plt.imshow(output2[:, :, 0])
    plt.title('Probability map of land (and water)')

    plt.subplot(1, 3, 2)
    plt.imshow(output2[:, :, 1])
    plt.title('Probability map of cloud')
    
    plt.subplot(1, 3, 3)
    plt.imshow(output2[:, :, 2])
    plt.title('Probability map of shadows')

    plt.show()