import numpy as np
import cldetector as cld
import matplotlib.pyplot as plt

if __name__ == '__main__':
    'Loading data'
    test_set = np.load('test/test_set_pixelwise.npz')
    input = test_set['arr_0']
    
    'Predicting'
    glassball = cld.CloudDetector(input, modelType='pixel')
    output = glassball.predict()
    
    'Show results'
    plt.figure(figsize=(12,5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(output[:, :, 0])
    plt.title('Probability map of land (and water)')

    plt.subplot(1, 3, 2)
    plt.imshow(output[:, :, 1])
    plt.title('Probability map of cloud')
    
    plt.subplot(1, 3, 3)
    plt.imshow(output[:, :, 2])
    plt.title('Probability map of shadows')

    plt.show()