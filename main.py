import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics, Input
from tensorflow.keras.utils import plot_model
from sklearn import metrics as sklearn_metrics

SAMPLES = 500000 # per class
N_VALIDATION = 10000
N_TESTING = 10000

################################### 1 Data ###################################

# Download from https://www.kaggle.com/datasets/alikookani/quantangle1
data = pd.read_csv('data/data.csv')

# Convert the DataFrame to a NumPy array for further processing
data = np.array(data)

# Separating real and imaginary values
vector = np.empty([len(data), 32], dtype=float)
for i in range(len(data)):
    for j in range(16):
        vector[i, 2 * j] = complex(data[i, j]).real
        vector[i, 2 * j + 1] = complex(data[i, j]).imag

# Reshaping to 3D tensors
tensor = vector.reshape(len(data), 4, 4, 2)

# Shuffle the dataset
tensor_shuffled = np.empty([len(data), 4, 4, 2], dtype=float)
class_label = np.ones(len(data)).astype('int')
for i in range(SAMPLES):
    tensor_shuffled[2 * i, :, :, :] = tensor[i, :, :, :]
    tensor_shuffled[2 * i + 1, :, :, :] = tensor[i + SAMPLES, :, :, :]
    class_label[2 * i] = 0

################################### 2 Model ###################################

input_tensor = Input(shape=(4, 4, 2))
layer1 = layers.Conv2D(64, 1)(input_tensor)
layer1 = layers.LeakyReLU()(layer1)
layer2 = layers.Conv2D(64, 1)(layer1)
layer2 = layers.LeakyReLU()(layer2)
branch_a = layers.Conv2D(256, 4)(layer2)
branch_a = layers.LeakyReLU()(branch_a)
branch_a = layers.Conv2D(128, 1)(branch_a)
branch_a = layers.LeakyReLU()(branch_a)
branch_b = layers.Conv2D(192, 3)(layer2)
branch_b = layers.LeakyReLU()(branch_b)
layer3 = layers.Conv2D(128, 2)(layer2)
layer3 = layers.LeakyReLU()(layer3)
layer4 = layers.Conv2D(128, 1)(layer3)
layer4 = layers.LeakyReLU()(layer4)
branch_c = layers.Conv2D(96, 3)(layer4)
branch_c = layers.LeakyReLU()(branch_c)
layer5 = layers.Conv2D(128, 2)(layer4)
layer5 = layers.LeakyReLU()(layer5)
layer5 = layers.concatenate([layer5, branch_b], axis=-1)
layer6 = layers.Conv2D(256, 1)(layer5)
layer6 = layers.LeakyReLU()(layer6)
layer7 = layers.Conv2D(256, 2)(layer6)
layer7 = layers.LeakyReLU()(layer7)
layer7 = layers.concatenate([layer7, branch_a], axis=-1)
layer8 = layers.Conv2D(512, 1)(layer7)
layer8 = layers.LeakyReLU()(layer8)
layer8 = layers.concatenate([layer8, branch_c], axis=-1)
layer9 = layers.Conv2D(512, 1)(layer8)
layer9 = layers.LeakyReLU()(layer9)
layer10 = layers.Conv2D(512, 1)(layer9)
layer10 = layers.LeakyReLU()(layer10)
top1 = layers.Flatten()(layer10)
top1 = layers.Dense(512)(top1)
top1 = layers.LeakyReLU()(top1)
top3 = layers.Dense(1)(top1)
top3 = layers.Activation('sigmoid', name='classification')(top3)
model = models.Model(input_tensor, top3)

model.summary()

################################### 2 Training ###################################

callback_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='my_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,          # LR *= 0.1 bei Plateau (z.B. 0.01 -> 0.001 -> 0.0001)
        patience=2,          # wie viele Epochen ohne Verbesserung warten
        min_lr=1e-6,         # nicht kleiner als das
        verbose=1
    ),
    # optional, aber oft sinnvoll:
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss={'classification': losses.binary_crossentropy},
    metrics={'classification': [metrics.binary_accuracy]},
)

history = model.fit(
    tensor_shuffled[:len(data)-N_VALIDATION-N_TESTING, :, :, :],
    {'classification': class_label[:len(data)-N_VALIDATION-N_TESTING]},
    batch_size=125,
    epochs=20,
    callbacks=callback_list,
    validation_data=(
        tensor_shuffled[len(data)-N_VALIDATION-N_TESTING:len(data)-N_TESTING, :, :, :],
        {'classification': class_label[len(data)-N_VALIDATION-N_TESTING:len(data)-N_TESTING]}
    ),
)

################################### 3 Evaluation ###################################

# Load the best saved model
my_model = tf.keras.models.load_model("my_model.keras")

# Evaluate the model using TensorFlow/Keras metrics
my_model.evaluate(tensor_shuffled[len(data)-N_TESTING:len(data), :, :, :], {'classification': class_label[len(data)-N_TESTING:len(data)]})

# Make predictions and create a confusion matrix
actual = class_label[len(data)-N_TESTING:len(data)]
seq_predictions = my_model.predict(tensor_shuffled[len(data)-N_TESTING:len(data), :, :])
seq_predictions = np.transpose(seq_predictions)[0]
seq_predictions = list(map(lambda x: 0 if x < 0.5 else 1, seq_predictions))

# Calculate the confusion matrix using sklearn metrics
confusion_matrix = sklearn_metrics.confusion_matrix(actual, seq_predictions)

# Display the confusion matrix
cm_display = sklearn_metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.show()