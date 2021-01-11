import tensorflow as tf
class FIIPDiscriminator(tf.keras.Model):

  def __init__(self,input_shape):
    super(FIIPDiscriminator, self).__init__()
    self.inshape = input_shape
    self.conv1  = tf.keras.layers.Conv2D(40,(10,10),activation=tf.nn.tanh,input_shape = input_shape)
    self.conv2  = tf.keras.layers.Conv2D(40,(10,10),activation=tf.nn.tanh)
    self.flat1   = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(150, activation=tf.nn.tanh)
    self.dense2 = tf.keras.layers.Dense(70, activation=tf.nn.tanh)
    self.dense3 = tf.keras.layers.Dense(30, activation=tf.nn.tanh)
    self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

  def call(self, inputs):
    x1 = self.conv2(self.conv1(inputs))
    x2 = self.flat1(x1)
    return self.dense4(self.dense3(self.dense2(self.dense1(x2))))

  def summary(self):
    x = tf.keras.Input(shape=self.inshape)
    model = tf.keras.Model(inputs=[x], outputs=self.call(x))
    return model.summary()