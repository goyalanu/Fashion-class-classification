#importing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fashion_train_df = pd.read_csv('fashion-mnist_train.csv', sep = ',')
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep = ',')
