#importing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fashion_train_df = pd.read_csv('fashion-mnist_train.csv', sep = ',')
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep = ',')

#visualizing the dataset
fashion_train_df.head()
fashion_train_df.tail()

fashion_train_df.shape
fashion_test_df.shape

training = np.array(fashion_train_df, dtype = 'float32')
testing = np.array(fashion_test_df, dtype = 'float32')
import random
i = random.randint(1, 6000)
plt.imshow(training[i, 1:].reshape(28, 28))
label = training[i, 0]

W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))

axes = axes.ravel()

n_training = len(training)

for i in np.arange(0, W_grid * L_grid):
    
    index = np.random.randint(0, n_training)
    
    axes[i].imshow( training[index,1:].reshape((28,28)))
    axes[i].set_title(training[index,0], fontsize = 8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)

#training the model
X_train = training[:, 1:]/255
y_train = training[:, 0]

X_test = testing[:, 1:]/255
y_test = testing[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 12345)
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))
X_train.shape
X_test.shape
X_validate.shape

import keras
from keras.model import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model = Sequential()

cnn_model.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

cnn_model.add(MaxPooling2D(pool_size = (2,2)))
cnn_model.add(Flatter())
cnn_model.add(Dense(output_dim = 32, activation = 'relu'))
cnn_model.add(Dense(output_dim = 32, activation = 'sigmoid'))
cnn_model.compile(loss ='asparse_categorical_crossentropy', optimizersadam(lr=0.001), matrics =['accuracy'])
epochs = 50
cnn_model.fix(X_train, y_train,batch_size = 512, nb_epoch = epochs, verbose = 1, validation_data = (X_validate, y_validate))

#evaluating the model
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
predicted_classes = cnn_model.predict_classes(X_test)
predicted_classes

L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction classes = (:0.1f)\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')
    
plt.subplots_adjust(wspace=0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)

from sklearn.matrics import classification_report

num_classes = 10
target_names = ["Class ()".format(i) for i in range(num_classes)]

print(classification_report(y_test, predicted_classes, target_names = target_names))
