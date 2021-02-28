import tensorflow as tf
from tensorflow.keras.layers import Conv2D,ReLU,BatchNormalization,Flatten,Dense,Conv2DTranspose,Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

net_input_dim = (552, 41,1)

def create_generator():
    generator = Sequential(name="G")
    
    # 8x8x128
    generator.add(Conv2D(20, (20, 20), activation='tanh', strides=1, padding='same', kernel_initializer=RandomNormal(0, 0.002),input_shape=net_input_dim,name="1"))

    # 8x8x128
    generator.add(Conv2D(20, (20, 20), activation='tanh', strides=1, padding='same', kernel_initializer=RandomNormal(0, 0.002),name="2"))

    # 8x8x128
    generator.add(Conv2D(20, (20, 20), activation='tanh', strides=1, padding='same', kernel_initializer=RandomNormal(0, 0.002),name="3"))
    
    generator.add(Flatten()) 
    generator.add(Dense(100, activation='tanh',name="7"))
    generator.add(Dense(500, activation='tanh',name="8"))
    generator.add(Dense(500, activation='tanh',name="9"))
    generator.add(Dense(500, activation='tanh',name="10"))
    generator.add(Dense(2829, activation='tanh',name="11"))
    generator.add(Reshape((69,41,1)))

    generator.add(Conv2DTranspose(20, (10, 10), activation='tanh', strides=(2,1), padding='same', kernel_initializer=RandomNormal(0, 0.01),name="12"))

    generator.add(Conv2DTranspose(20, (10, 10), activation='tanh', strides=(2,1), padding='same', kernel_initializer=RandomNormal(0, 0.01),name="13"))

    generator.add(Conv2DTranspose(1, (10, 10), activation='tanh', strides=(2,1), padding='same', kernel_initializer=RandomNormal(0, 0.01),name="14"))

    generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return generator

def create_discriminator():
    discriminator = Sequential(name="D")
    
    discriminator.add(Conv2D(40, (20, 20), activation=tf.nn.relu, kernel_initializer=RandomNormal(0, 0.02), input_shape=net_input_dim,name="5"))
    
    discriminator.add(Conv2D(40, (20, 20), activation=tf.nn.relu, kernel_initializer=RandomNormal(0, 0.02),name="6"))
    
    discriminator.add(BatchNormalization())
    discriminator.add(Flatten())

    discriminator.add(Dense(90, activation='tanh',name="8"))
    discriminator.add(Dense(50, activation='tanh',name="9"))
    discriminator.add(Dense(1, activation='sigmoid',name="10"))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    return discriminator
class FIIPGenerator(tf.keras.Model):

  def __init__(self,input_shape):
    super(FIIPGenerator, self).__init__()
    self.inshape = input_shape
    self.conv1  = tf.keras.layers.Conv2D(40,(10,10),activation=tf.nn.relu,input_shape = input_shape)
    self.conv2  = tf.keras.layers.Conv2D(40,(10,10),activation=tf.nn.relu)
    self.conv3  = tf.keras.layers.Conv2D(40,(10,10),activation=tf.nn.relu)
    self.flat1   = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
    self.dense2 = tf.keras.layers.Dense(200, activation=tf.nn.tanh)
    self.dense3 = tf.keras.layers.Dense(22632, activation=tf.nn.relu)
    self.flat2   = tf.keras.layers.Reshape((552,41,1))

  def call(self, inputs):
    x1 = self.conv1(inputs)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    x4 = self.flat1(x3)
    x5 = self.dense1(x4)
    x6 = self.dense2(x5)
    x7 = self.dense3(x6)
    x8 = self.flat2(x7)
    output = tf.keras.layers.add([inputs,x8])
    return output

  def summary(self):
    x = tf.keras.Input(shape=self.inshape)
    model = tf.keras.Model(inputs=[x], outputs=self.call(x))
    return model.summary()
    
class FIIPDiscriminator(tf.keras.Model):

  def __init__(self,input_shape):
    super(FIIPDiscriminator, self).__init__()
    self.inshape = input_shape
    self.conv1  = tf.keras.layers.Conv2D(40,(10,10),activation=tf.nn.relu,input_shape = input_shape)
    self.conv2  = tf.keras.layers.Conv2D(40,(10,10))
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    self.flat1   = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(150, activation=tf.nn.tanh)
    self.dense2 = tf.keras.layers.Dense(70, activation=tf.nn.tanh)
    self.dense3 = tf.keras.layers.Dense(30, activation=tf.nn.tanh)
    self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

  def call(self, inputs):
    x1 = self.conv2(self.conv1(inputs))
    x2 = self.batchnorm1(x1)
    x3 = tf.keras.activations.tanh(x2)
    x4 = self.flat1(x3)
    return self.dense4(self.dense3(self.dense2(self.dense1(x4))))

  def summary(self):
    x = tf.keras.Input(shape=self.inshape)
    model = tf.keras.Model(inputs=[x], outputs=self.call(x))
    return model.summary()