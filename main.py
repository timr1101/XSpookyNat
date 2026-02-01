import numpy as np
import qiskit as qi
import qutip as Q
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics, Input
from tensorflow.keras.utils import plot_model
from sklearn import metrics as sklearn_metrics


# Set the number of samples
Samples = 500000

# Initialize arrays for entangled and separable states
Entangled = np.empty([Samples, 16], dtype=complex)
Separable = np.empty([Samples, 16], dtype=complex)

# Initialize arrays for labels
ent_label = np.empty([Samples, 2], dtype=float)
sep_label = np.empty([Samples, 2], dtype=float)

# Import necessary libraries for file upload and data manipulation
import pandas as pd
import numpy as np

# Read the data from a CSV file ('mydata.csv') into a Pandas DataFrame
data = pd.read_csv('/content/train/mydata.csv')

# Convert the DataFrame to a NumPy array for further processing
data = np.array(data)

# Start measuring execution time (It takes half an hour or so!)
start = time.time()

# Flags for determining each state's number
i = 0
j = 0

while i < Samples or j < Samples:
    Rho = Q.rand_dm(4, dims=[[2, 2], [2, 2]])
    Density = np.array(Rho)
    temp1 = qi.quantum_info.entanglement_of_formation(Density)

    if temp1 == 0.0 and i < Samples:
        Separable[i, :16] = Density.reshape(1, 16)
        sep_label[i, 0] = temp1
        sep_label[i, 1] = np.linalg.det(Q.partial_transpose(Rho, [0, 1]).full()).real
        i += 1
    elif j < Samples:
        Entangled[j, :16] = Density.reshape(1, 16)
        ent_label[j, 0] = temp1
        ent_label[j, 1] = np.linalg.det(Q.partial_transpose(Rho, [0, 1]).full()).real
        j += 1

# End measuring execution time
end = time.time()
print("Execution time:", end - start)

# Combine data and labels
temp2 = np.concatenate((Separable, sep_label), axis=1)
temp3 = np.concatenate((Entangled, ent_label), axis=1)
data = np.concatenate((temp2, temp3), axis=0)
del temp1, temp2, temp3

# Separating real and imaginary values
vector = np.empty([len(data), 32], dtype=float)
for i in range(len(data)):
    for j in range(16):
        vector[i, 2 * j] = complex(data[i, j]).real
        vector[i, 2 * j + 1] = complex(data[i, j]).imag
# Reshaping to 3D tensors
matrix = vector.reshape(len(data), 4, 8)
tensor1 = matrix.reshape(len(data), 4, 4, 2)

# Shuffle the dataset
tensor2 = np.empty([len(data), 4, 4, 2], dtype=float)
reg_label = np.empty([len(data)], dtype=float)
class_label = np.ones(len(data)).astype('int')
for i in range(Samples):
    tensor2[2 * i, :, :, :] = tensor1[i, :, :, :]
    reg_label[2 * i] = complex(data[i, 17]).real
    class_label[2 * i] = 0
    tensor2[2 * i + 1, :, :, :] = tensor1[i + Samples, :, :, :]
    reg_label[2 * i + 1] = complex(data[i + Samples, 17]).real

#XpookyNet Model

input_tensor = Input(shape=(4, 4, 2))
layer1 = layers.Conv2D(64, 1)(input_tensor)
#layer1 = layers.BatchNormalization()(layer1)
layer1 = layers.LeakyReLU()(layer1)
layer2 = layers.Conv2D(64, 1)(layer1)
#layer2 = layers.BatchNormalization()(layer2)
layer2 = layers.LeakyReLU()(layer2)
branch_a = layers.Conv2D(256, 4)(layer2)
#branch_a = layers.BatchNormalization()(branch_a)
branch_a = layers.LeakyReLU()(branch_a)
branch_a = layers.Conv2D(128, 1)(branch_a)
#branch_a = layers.BatchNormalization()(branch_a)
branch_a = layers.LeakyReLU()(branch_a)
branch_b = layers.Conv2D(192, 3)(layer2)
#branch_b = layers.BatchNormalization()(branch_b)
branch_b = layers.LeakyReLU()(branch_b)
layer3 = layers.Conv2D(128, 2)(layer2)
#layer3 = layers.BatchNormalization()(layer3)
layer3 = layers.LeakyReLU()(layer3)
layer4 = layers.Conv2D(128, 1)(layer3)
#layer4 = layers.BatchNormalization()(layer4)
layer4 = layers.LeakyReLU()(layer4)
branch_c = layers.Conv2D(96, 3)(layer4)
#branch_c = layers.BatchNormalization()(branch_c)
branch_c = layers.LeakyReLU()(branch_c)
layer5 = layers.Conv2D(128, 2)(layer4)
#layer5 = layers.BatchNormalization()(layer5)
layer5 = layers.LeakyReLU()(layer5)
layer5 = layers.concatenate([layer5, branch_b], axis=-1)
layer6 = layers.Conv2D(256, 1)(layer5)
#layer6 = layers.BatchNormalization()(layer6)
layer6 = layers.LeakyReLU()(layer6)
layer7 = layers.Conv2D(256, 2)(layer6)
#layer7 = layers.BatchNormalization()(layer7)
layer7 = layers.LeakyReLU()(layer7)
layer7 = layers.concatenate([layer7, branch_a], axis=-1)
layer8 = layers.Conv2D(512, 1)(layer7)
#layer8 = layers.BatchNormalization()(layer8)
layer8 = layers.LeakyReLU()(layer8)
layer8 = layers.concatenate([layer8, branch_c], axis=-1)
layer9 = layers.Conv2D(512, 1)(layer8)
#layer9 = layers.BatchNormalization()(layer9)
layer9 = layers.LeakyReLU()(layer9)
layer10 = layers.Conv2D(512, 1)(layer9)
#layer10 = layers.BatchNormalization()(layer10)
layer10 = layers.LeakyReLU()(layer10)
top1 = layers.Flatten()(layer10)
top1 = layers.Dense(512)(top1)
#top1 = layers.BatchNormalization()(top1)
top1 = layers.LeakyReLU()(top1)
#top2 = layers.Dense(1, name='regression')(top1)
top3 = layers.Dense(1)(top1)
#top3 = layers.BatchNormalization()(top3)
top3 = layers.Activation('sigmoid', name='classification')(top3)
#model = models.Model(input_tensor, [top2, top3])
model = models.Model(input_tensor, top3)
model.summary()

plot_model(model, show_shapes=True, to_file='model.png')

# Compile the model
callback_list = [
                 tf.keras.callbacks.ModelCheckpoint(
                     filepath = 'my_model.h5',
                     monitor = 'val_loss',
                     save_best_only = True,
                 )
]

model.compile(
    optimizer = optimizers.SGD(
        learning_rate = 0.01, momentum = 0.9,
    ),
    loss = {
#        'regression': losses.mse,
        'classification': losses.binary_crossentropy
        },
    metrics = {
#        'regression': metrics.mae,
        'classification': metrics.binary_accuracy
        }
)

history1 = model.fit(
    tensor2[:len(data)-20000, :, :, :],
    {#'regression': reg_label[:len(data)-20000],
     'classification': class_label[:len(data)-20000]},
    batch_size = 125,
    epochs = 8,
    callbacks = callback_list,
    validation_data = (tensor2[len(data)-20000:len(data)-10000, :, :, :],
                     {#'regression': reg_label[len(data)-20000:len(data)-10000],
                      'classification': class_label[len(data)-20000:len(data)-10000]}
                     ),
)

#First manual reduction in learning rate
model.compile(
    optimizer=optimizers.SGD(
        learning_rate=0.001, momentum=0.9,
    ),
    loss = {
#        'regression': losses.mse,
        'classification': losses.binary_crossentropy
        },
    metrics = {
#        'regression': metrics.mae,
        'classification': metrics.binary_accuracy
        }
)

history2 = model.fit(
    tensor2[:len(data)-20000, :, :, :],
    {#'regression': reg_label[:len(data)-20000],
     'classification': class_label[:len(data)-20000]},
    batch_size = 125,
    epochs = 4,
    callbacks = callback_list,
    validation_data = (tensor2[len(data)-20000:len(data)-10000, :, :, :],
                     {#'regression': reg_label[len(data)-20000:len(data)-10000],
                      'classification': class_label[len(data)-20000:len(data)-10000]}
                     ),
)

#Second manual reduction in learning rate
model.compile(
    optimizer=optimizers.SGD(
        learning_rate=0.0001, momentum=0.9,
    ),
    loss = {
#        'regression': losses.mse,
        'classification': losses.binary_crossentropy
        },
    metrics = {
#        'regression': metrics.mae,
        'classification': metrics.binary_accuracy
        }
)

history3 = model.fit(
    tensor2[:len(data)-20000, :, :, :],
    {#'regression': reg_label[:len(data)-20000],
     'classification': class_label[:len(data)-20000]},
    batch_size = 125,
    epochs = 2,
    callbacks = callback_list,
    validation_data = (tensor2[len(data)-20000:len(data)-10000, :, :, :],
                     {#'regression': reg_label[len(data)-20000:len(data)-10000],
                      'classification': class_label[len(data)-20000:len(data)-10000]}
                     ),
)


# Load the best saved model
my_model = tf.keras.models.load_model("my_model.h5")

# Evaluate the model using TensorFlow/Keras metrics
my_model.evaluate(tensor2[len(data)-10000:len(data), :, :, :], {#'regression': reg_label[len(data)-10000:len(data)],
                                                     'classification': class_label[len(data)-10000:len(data)]})

# Make predictions and create a confusion matrix
actual = class_label[990000:1000000]
seq_predictions = my_model.predict(tensor2[990000:1000000, :, :])
seq_predictions = np.transpose(seq_predictions)[0]
seq_predictions = list(map(lambda x: 0 if x < 0.5 else 1, seq_predictions))

# Calculate the confusion matrix using sklearn metrics
confusion_matrix = sklearn_metrics.confusion_matrix(actual, seq_predictions)

# Display the confusion matrix
cm_display = sklearn_metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.show()