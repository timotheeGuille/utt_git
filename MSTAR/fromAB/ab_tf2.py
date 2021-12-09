import tensoflow as tf


def netG(z, y, BATCH_SIZE):
    
   # concat attribute y onto z
   z = tf.concat([z,y], axis=1)
   z = tcl.fully_connected(z, 4*4*512, activation_fn=tf.identity, scope='g_z')
   #z = tcl.batch_norm(z)
   z = tf.reshape(z, [BATCH_SIZE, 4, 4, 512])
   #z = tf.nn.relu(z)

   conv1 = tcl.convolution2d_transpose(z, 512, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   
   conv2 = tcl.convolution2d_transpose(conv1, 256, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   
   conv3 = tcl.convolution2d_transpose(conv2, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')

   conv4 = tcl.convolution2d_transpose(conv3, 64, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv4')

   conv5 = tcl.convolution2d_transpose(conv4, 1, 5, 2, activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv5')



   print('z:',z)
   print('g_conv1:',conv1)
   print('g_conv2:',conv2)
   print('g_conv3:',conv3)
   print('g_conv4:',conv4)
   print('g_conv5:',conv5)
   print('END G')
   tf.add_to_collection('vars', z)
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)
   return conv5