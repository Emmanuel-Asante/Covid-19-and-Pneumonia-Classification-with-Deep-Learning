# Import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

import numpy
from sklearn.metrics import classification_report, confusion_matrix

# Create an ImageDataGenerator object to load image data
data_generator = ImageDataGenerator(
  rescale=1.0/255,
  zoom_range=0.2,
  rotation_range=30,
  width_shift_range=0.05,
  height_shift_range=0.05
)

# Create a training data iterator
training_iterator = data_generator.flow_from_directory("augmented-data/train", class_mode="categorical", color_mode="grayscale", target_size=(256, 256), batch_size=8)

# Create a validation data iterator
validation_iterator = data_generator.flow_from_directory("augmented-data/test", class_mode="categorical", color_mode="grayscale", target_size=(256, 256), batch_size=8)

# Create a neural network model
model = Sequential()

# Add input layer to the model
model.add(tf.keras.Input(shape=(256, 256, 1)))

# Add a Conv2D layer with 8 filters each size 3x3, and stride of 2
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, padding="valid", activation="relu"))

# Add a max pooling layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

# Add a Conv2D layer with 8 filters each size 3x3, and stride of 2
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, padding="valid", activation="relu"))

# Add a max pooling layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

# Add a Flatten layer
model.add(tf.keras.layers.Flatten())

# Add hidden Dense Layer with 64 hidden units
model.add(tf.keras.layers.Dense(64, activation="relu"))

# Add output Dense Layer
model.add(tf.keras.layers.Dense(3, activation="softmax"))

# Print out the model's summary
print(model.summary())

# Compile the model
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

# Train the model on training_iterator
history = model.fit(training_iterator, steps_per_epoch=training_iterator.samples/8, epochs=8, validation_data=validation_iterator, validation_steps=validation_iterator.samples/8)

# Do Matplotlib extension below

# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping
fig.tight_layout()

# use this savefig call at the end of your graph instead of using plt.show()
plt.savefig('static/images/my_plots.png')

# Calculate test steps per epoch
test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)

# Make prediction using validation_iterator
predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)

# Calculate test steps per epoch
test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)

# Return indices of maximum predicted values
predicted_classes = numpy.argmax(predictions, axis=1)

# Return true classes
true_classes = validation_iterator.classes

# List of class labels
class_labels = list(validation_iterator.class_indices.keys())

# Calculate calssification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)

# Print out report
print(report)   
 
# Calculate the confusion_matrix
cm=confusion_matrix(true_classes,predicted_classes)

# Print out cm
print(cm)