import tensorflow as tf

class CloudDetector():
    'Predict clouds'
    def __init__(self, input):
        'Initialization'
        self.input = input
        self.input_max = 1.2749
        self.loadModel()
        
    def updateInput(self, input):
        'Return prediction'  
        self.input = input
        
    def predict(self):
        'Return prediction'        
        return self.model.predict(self.input)

    def loadModel(self):
        max_norm_rate = tf.keras.constraints.MaxNorm(1)
        l1_rate = 0.0
        l2_rate = 0.0
        dropout_rate = 0.5

        self.model = tf.keras.Sequential(name="pixelwise_v8")

        self.model.add(tf.keras.layers.Lambda(lambda x: (x / self.input_max) - 0.5))

        self.model.add(tf.keras.layers.Dense(512, activation="relu", kernel_constraint=max_norm_rate))
        self.model.add(tf.keras.layers.ActivityRegularization(l1=l1_rate, l2=l2_rate))
        self.model.add(tf.keras.layers.GaussianDropout(dropout_rate))

        self.model.add(tf.keras.layers.Dense(512, activation="relu", kernel_constraint=max_norm_rate))
        self.model.add(tf.keras.layers.ActivityRegularization(l1=l1_rate, l2=l2_rate))
        self.model.add(tf.keras.layers.GaussianDropout(dropout_rate))

        self.model.add(tf.keras.layers.Dense(256, activation="relu", kernel_constraint=max_norm_rate))
        self.model.add(tf.keras.layers.ActivityRegularization(l1=l1_rate, l2=l2_rate))
        self.model.add(tf.keras.layers.GaussianDropout(dropout_rate))

        self.model.add(tf.keras.layers.Dense(256, activation="relu", kernel_constraint=max_norm_rate))
        self.model.add(tf.keras.layers.ActivityRegularization(l1=l1_rate, l2=l2_rate))
        self.model.add(tf.keras.layers.GaussianDropout(dropout_rate))

        self.model.add(tf.keras.layers.Dense(128, activation="relu", kernel_constraint=max_norm_rate))
        self.model.add(tf.keras.layers.ActivityRegularization(l1=l1_rate, l2=l2_rate))
        self.model.add(tf.keras.layers.Dropout(dropout_rate))

        self.model.add(tf.keras.layers.Dense(128, activation="relu", kernel_constraint=max_norm_rate))
        self.model.add(tf.keras.layers.ActivityRegularization(l1=l1_rate, l2=l2_rate))
        self.model.add(tf.keras.layers.Dropout(dropout_rate))

        self.model.add(tf.keras.layers.Dense(64, activation="relu"))

        self.model.add(tf.keras.layers.Dense(16, activation="relu"))

        self.model.add(tf.keras.layers.Dense(3, activation="softmax"))
                       
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        self.model.load_weights("pixelwise_v8_weights.hdf5")