import tensorflow as tf
class FIIPDiscriminator(tf.keras.Model):

  def __init__(self):
    super(FIIPDiscriminator, self).__init__()
    self.conv1  = tf.keras.layers.Conv1D(50,2,activation=tf.nn.tanh,input_shape = (1102,2))
    self.flat1   = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(70, activation=tf.nn.tanh)
    self.dense2 = tf.keras.layers.Dense(50, activation=tf.nn.tanh)
    self.dense3 = tf.keras.layers.Dense(10, activation=tf.nn.tanh)
    self.dense4 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

  def call(self, inputs):
    x1 = self.conv1(inputs)
    x2 = self.flat1(x1)
    return self.dense4(self.dense3(self.dense2(self.dense1(x2))))

  def summary(self):
    x = tf.keras.Input(shape=(1102,2))
    model = tf.keras.Model(inputs=[x], outputs=self.call(x))
    return model.summary()