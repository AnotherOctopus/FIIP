import tensorflow as tf
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
    self.dense3 = tf.keras.layers.Dense(22632, activation=tf.nn.tanh)
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