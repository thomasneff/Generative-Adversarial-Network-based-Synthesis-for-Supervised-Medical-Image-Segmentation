import tensorflow as tf
import os


def string_tensor_from_idlist_and_path(idlist_path, folder_path, name = None):
  with open(idlist_path) as f:
    file_names = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  file_names = [folder_path + x.strip() for x in file_names] 
  
  string_tensor = tf.Variable(file_names, trainable=False,
                              name=name, validate_shape=False)
  return string_tensor
