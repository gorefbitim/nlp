import numpy as np
import keras

class CustomGenerator(keras.utils.Sequence) :
  
  def __init__(self, X_train_indices, y_train, batch_size = 32) :
    self.x = X_train_indices
    self.y = y_train
    self.batch_size = batch_size
    
  def __len__(self) :
    return (np.ceil(len(self.x) / float(self.batch_size))).astype(np.int)
    
  def __getitem__(self, idx) :
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x , batch_y
