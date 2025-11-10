import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)

# 2. DEFINE DATA LOADER

def load_cifar10_batch(file):
   """Load a single batch from CIFAR-10 local binary files."""
   with open(file, 'rb') as f:
       batch = pickle.load(f, encoding='bytes')
   data = batch[b'data']
   labels = batch[b'labels']
   data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
   # FIX: Ensure labels are returned as a NumPy array immediately
   return data, np.array(labels) 


# 3. LOAD LOCAL CIFAR-10 DATASET

x_train_list = []
y_train_list = [] # Use a list for accumulating labels
for i in range(1, 6):
   data_batch, labels_batch = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
   x_train_list.append(data_batch)
   y_train_list.append(labels_batch) # Append the NumPy array batches

x_train = np.concatenate(x_train_list)
# FIX: Concatenate the list of NumPy arrays for labels
y_train = np.concatenate(y_train_list) 

# --- Load test batch ---
# The load_cifar10_batch function now returns y_test as a NumPy array
x_test, y_test = load_cifar10_batch('cifar-10-batches-py/test_batch') 

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Ensure y_test is explicitly a NumPy array (good practice, though fixed above)
y_test = np.asarray(y_test)

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)