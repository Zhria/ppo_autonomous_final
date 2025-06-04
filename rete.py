#File per definire una rete neurale dati i numero di input e di output.
import tensorflow as tf
from keras import layers

class Actor(tf.keras.Model):
    def __init__(self,name, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.name=name
        self.cnn=tf.keras.Sequential([
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform", input_shape=(64,64,3), name=self.name+"_conv1"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool1"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv2"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool2"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv3"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool3"),
                layers.Flatten(data_format="channels_last",name=self.name+"_flatten"),
                layers.Dense(128, activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_dense"),   
            ])
        self.lastLayer=layers.Dense(15, kernel_initializer="glorot_uniform", bias_initializer="zeros",name=self.name+"_dense4")
    def call(self, inputs):
        #if inputs has 64,64,3 shape then it's an image and we need to reshape it to 1,64,64,3
        if len(inputs.shape)==3:
            inputs=tf.reshape(inputs,[1,inputs.shape[0],inputs.shape[1],inputs.shape[2]])


        #Controllo che l'input sia un tensore.
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)

        #input normalization
        inputs = tf.cast(inputs, tf.float32) / 255.0
        
        x=self.cnn.call(inputs)
        x=self.lastLayer(x)
        return x

class Critic(tf.keras.Model):
    def __init__(self,name,**kwargs):
        super(Critic, self).__init__(**kwargs)
        self.name=name
        self.cnn=tf.keras.Sequential([
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform", input_shape=(64,64,3), name=self.name+"_conv1"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool1"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv2"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool2"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv3"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool3"),
                layers.Flatten(data_format="channels_last",name=self.name+"_flatten"),
                layers.Dense(128, activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_dense"),
            ])
        self.lastLayer=layers.Dense(1, kernel_initializer="glorot_uniform", bias_initializer="zeros",name=self.name+"_dense4")
    
    def call(self, inputs):
        #if inputs has 64,64,3 shape then it's an image and we need to reshape it to 1,64,64,3
        if len(inputs.shape)==3:
            inputs=tf.reshape(inputs,[1,inputs.shape[0],inputs.shape[1],inputs.shape[2]])

        #Controllo che l'input sia un tensore.
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)
        #input normalization
        inputs = tf.cast(inputs, tf.float32) / 255.0
        x=self.cnn.call(inputs)
        x=self.lastLayer(x)
        return x