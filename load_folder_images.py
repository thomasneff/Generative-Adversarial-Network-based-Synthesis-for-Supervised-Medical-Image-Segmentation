import tensorflow as tf


def _decode_and_preprocess_image(image_file, shift_param = -128, rescale_param = 128, resized_image_size = [128, 128]):
  image = tf.image.decode_png(image_file);
  
  #image.set_shape((32, 32, 3))
  
  
  image = tf.cast(image, tf.float32)
  
  if shift_param != 0:
    image = tf.add(image, shift_param)
  
  if rescale_param != 0:
    image = tf.multiply(image, 1.0/rescale_param)
  
  
  
  #image = tf.random_crop(image, [32, 32, 3])
  image = tf.image.resize_images(image, resized_image_size) 
  image.set_shape((resized_image_size[0], resized_image_size[1], 1))
  #image = tf.image.grayscasle_to_rgb(image, 'torgb')
  return image


def _load_images(image_file, batch_size, num_preprocess_threads, min_queue_examples, shift_param = -128, rescale_param = 128, resized_image_size = [128, 128], shuffle = True):
  
  image = _decode_and_preprocess_image(image_file, shift_param, rescale_param, resized_image_size)
  #image = tf.image.grayscasle_to_rgb(image, 'torgb')
  images = []
  if shuffle == True:
    images = tf.train.shuffle_batch(
          [image],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
  else:
    images = tf.train.batch(
          [image],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)
  
  
  return images

def load_image_and_segmentation_from_idlist(idlist_tensor_img, idlist_tensor_seg, batch_size, num_preprocess_threads, min_queue_examples, shift_params = [-128, -0.5], rescale_params = [128, 0.5], resized_image_size = [128, 128], shuffle = True):
  # Make a queue of file names including all the image files in the relative
  # image directory.
  filename_queue_image  = tf.train.string_input_producer(idlist_tensor_img,
                                                           shuffle=False)
  
  # Read an entire image file. If the images
  # are too large they could be split in advance to smaller files or use the Fixed
  # reader to split up the file.
  image_reader = tf.WholeFileReader()
  
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file_image = image_reader.read(filename_queue_image)
  
  processed_image = _decode_and_preprocess_image(image_file_image, shift_params[0], rescale_params[0], resized_image_size)
  
  # Make a queue of file names including all the image files in the relative
  # image directory.
  filename_queue_seg  = tf.train.string_input_producer(idlist_tensor_seg,
                                                         shuffle=False)
  
  
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file_seg = image_reader.read(filename_queue_seg)
  
  processed_seg = _decode_and_preprocess_image(image_file_seg, shift_params[1], rescale_params[1], resized_image_size)
  
  if shuffle == True:
    images, segmentations = tf.train.shuffle_batch(
          [processed_image, processed_seg],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
  else:
    images, segmentations = tf.train.batch(
          [processed_image, processed_seg],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)
    
  return images, segmentations
  

def load_images_from_idlist(idlist, batch_size, num_preprocess_threads, min_queue_examples, shift_param = -128, rescale_param = 128, resized_image_size = [128, 128], shuffle = True):
  # Make a queue of file names including all the image files in the relative
  # image directory.
  filename_queue = tf.train.string_input_producer(idlist,
                                                  shuffle=shuffle)
  
  # Read an entire image file. If the images
  # are too large they could be split in advance to smaller files or use the Fixed
  # reader to split up the file.
  image_reader = tf.WholeFileReader()
  
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file = image_reader.read(filename_queue)

  return _load_images(image_file, batch_size, num_preprocess_threads, min_queue_examples, shift_param, rescale_param, resized_image_size, shuffle)

def load_images(folder_path_match, batch_size, num_preprocess_threads, min_queue_examples, shift_param = -128, rescale_param = 128, resized_image_size = [128, 128], shuffle = True):
  # Make a queue of file names including all the image files in the relative
  # image directory.
  filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(folder_path_match),
    shuffle=shuffle)
  
  # Read an entire image file. If the images
  # are too large they could be split in advance to smaller files or use the Fixed
  # reader to split up the file.
  image_reader = tf.WholeFileReader()
  
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file = image_reader.read(filename_queue)

  return _load_images(image_file, batch_size, num_preprocess_threads, min_queue_examples, shift_param, rescale_param, resized_image_size, shuffle)

def load_image(image_path, num_preprocess_threads, min_queue_examples, resized_image_size = [64, 64]):
  # Make a queue of file names including all the image files in the relative
  # image directory.
  filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(image_path))
  
  # Read an entire image file. If the images
  # are too large they could be split in advance to smaller files or use the Fixed
  # reader to split up the file.
  image_reader = tf.WholeFileReader()
  
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file = image_reader.read(filename_queue)
  
  image = tf.image.decode_png(image_file);
  
  #image.set_shape((32, 32, 3))
  
  
  image = tf.cast(image, tf.float32)
  
  image = tf.add(image, -128)
  image = tf.multiply(image, 1.0/128)
  
  
  
  #image = tf.random_crop(image, [32, 32, 3])
  image = tf.image.resize_images(image, resized_image_size) 
  image.set_shape((resized_image_size[0], resized_image_size[1], 1))
  #image = tf.image.grayscasle_to_rgb(image, 'torgb')
  
  images = tf.train.shuffle_batch(
      [image],
      batch_size=1,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * 1,
      min_after_dequeue=min_queue_examples)
  
  
  
  return images