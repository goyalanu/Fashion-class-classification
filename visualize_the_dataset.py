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
