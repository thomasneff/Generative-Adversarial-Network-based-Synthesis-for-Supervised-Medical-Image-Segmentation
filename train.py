from datetime import datetime
import time

import tensorflow as tf

from dcgan import *
import numpy as np 
from scipy.misc import imsave
import load_folder_images
import os
from load_folder_images import load_image_and_segmentation_from_idlist
from load_folder_images import load_image
import csv
import util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/dcgan_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_string("working_directory", "working_dir", "")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "")




z_dim = 100
batch_size = 128
learning_rate = 0.0004
beta1 = 0.5
monitoring_batches = 200; #number of batches after which some samples are saved
num_monitoring_cycles = 100;




def train():

  idlist_img_name = "list_of_img_ids.txt"
  idlist_seg_name = "list_of_seg_ids.txt"
  img_folder_path = "image_folder_path/"
  seg_folder_path = "segmentation_folder_path/"

  idlist_tensor_img = util.string_tensor_from_idlist_and_path(idlist_img_name, img_folder_path)
  idlist_tensor_seg = util.string_tensor_from_idlist_and_path(idlist_seg_name, seg_folder_path)
  
  

  images, segmentation_images = load_image_and_segmentation_from_idlist(idlist_tensor_img, idlist_tensor_seg, batch_size, 16, 2560, shift_params = [-128, -0.5], rescale_params = [128, 0.5], shuffle = True)
  
  
  dcgan = DCGAN(batch_size)
  input_images = images
  
  input_plus_segmentation = tf.concat([input_images, segmentation_images], 3)
  
  train_op = dcgan.build(input_plus_segmentation)
  sample_images = dcgan.sample_images()
  
  saver = tf.train.Saver(max_to_keep = 50)
 

  
  with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)
      sess.run(tf.global_variables_initializer())
      
      
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      
      total_iters_max =  num_monitoring_cycles * monitoring_batches
      iters = 0
      total_duration = 0.0
      avg_iter_time = 0.0
      
      for cycle in xrange(num_monitoring_cycles):
        for batch in xrange(monitoring_batches):
          start_time = time.time()
          _, g_loss_value, d_loss_value = sess.run([train_op, dcgan.losses['g'], dcgan.losses['d']])
          _, g_loss_value, d_loss_value = sess.run([dcgan.g_opt_op, dcgan.losses['g'], dcgan.losses['d']])        
            
          duration = time.time() - start_time
          total_duration = total_duration + duration
          iters = iters + 1
          avg_iter_time = (avg_iter_time * (iters - 1) + duration) / iters
          #ETA_seconds = (total_iters_max / iters) * total_duration - total_duration
          ETA_seconds = (0.95 * avg_iter_time + 0.05 * duration) * (total_iters_max - iters)
          format_str = 'cycle (%d / %d), batch (%d / %d) loss = (G: %.8f, D: %.8f) (%.3f sec/batch) (ETA: %d seconds)'
          print(format_str % (cycle, num_monitoring_cycles, batch, monitoring_batches, g_loss_value, d_loss_value, duration, int(ETA_seconds)))
        
        
        checkpoint_folder = FLAGS.working_directory + "/" + FLAGS.checkpoint_dir + '/checkpoint%d.ckpt' % cycle
        if not os.path.exists(checkpoint_folder):
          os.makedirs(checkpoint_folder)
        saver.save(sess, checkpoint_folder)


        imgs = sess.run(sample_images)
         
        for k in range(batch_size):
            imgs_folder = os.path.join(FLAGS.working_directory, 'out/imgs%d/') % cycle
            if not os.path.exists(imgs_folder):
              os.makedirs(imgs_folder)
              
            segs_folder = os.path.join(FLAGS.working_directory, 'out/segs%d/') % cycle
            if not os.path.exists(segs_folder):
              os.makedirs(segs_folder)
              
            img_channel = imgs[k][:, :, 0]
            img_seg = imgs[k][:, :, 1]
            imsave(os.path.join(imgs_folder, 'img_%d.png') % k,
                     img_channel.reshape(128, 128))
            imsave(os.path.join(segs_folder, 'img_%d.png') % k,
                     img_seg.reshape(128, 128))
              
        imgs_in = sess.run(input_plus_segmentation)    
        for k in range(batch_size):
          imgs_folder = os.path.join(FLAGS.working_directory, 'in/imgs%d/') % cycle
          if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)
            
          segs_folder = os.path.join(FLAGS.working_directory, 'in/segs%d/') % cycle
          if not os.path.exists(segs_folder):
            os.makedirs(segs_folder)

          img_channel = imgs_in[k][:, :, 0]
          img_seg = imgs_in[k][:, :, 1]
          imsave(os.path.join(imgs_folder, 'img_%d.png') % k,
                     img_channel.reshape(128, 128))
          imsave(os.path.join(segs_folder, 'img_%d.png') % k,
                     img_seg.reshape(128, 128))
              
      coord.request_stop()
      coord.join(threads)
      
     

def main(argv=None):  # pylint: disable=unused-argument

  
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
