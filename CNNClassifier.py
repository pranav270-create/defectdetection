
import tensorflow as tf

class Classifier():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), activation=tf.keras.layers.LeakyReLU(0.05), input_shape=(80, 80, 3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.keras.layers.LeakyReLU(0.05), padding = 'same'),
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.keras.layers.LeakyReLU(0.05), padding = 'same'),
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.keras.layers.LeakyReLU(0.05), padding = 'same'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3,3), activation=tf.keras.layers.LeakyReLU(0.05), padding = 'same'),
    tf.keras.layers.Conv2D(128, (3,3), activation=tf.keras.layers.LeakyReLU(0.05), padding = 'same'),
    tf.keras.layers.Conv2D(128, (3,3), activation=tf.keras.layers.LeakyReLU(0.05), padding = 'same'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, (3,3), activation=tf.keras.layers.LeakyReLU(0.05)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])