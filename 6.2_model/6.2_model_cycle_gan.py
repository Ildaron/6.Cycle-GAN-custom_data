%tensorflow_version 1.x
 
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
 
class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
 
    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')
 
        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')
 
        self.input_spec = InputSpec(ndim=ndim)
 
        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)
 
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True
 
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))
 
        if self.axis is not None:
            del reduction_axes[self.axis]
 
        del reduction_axes[0]
 
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev
 
        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]
 
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed
 
    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
 
 
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
print ("ok") 
# define the discriminator model
def define_discriminator(image_shape):
        
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # source image input
  in_image = Input(shape=image_shape)
  # C64
  d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  # C128
  d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = InstanceNormalization(axis=-1)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # C256
  d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = InstanceNormalization(axis=-1)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # C512
  d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
  d = InstanceNormalization(axis=-1)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # second last output layer
  d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
  d = InstanceNormalization(axis=-1)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # patch output
  patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
  # define model
  model = Model(in_image, patch_out)
  # compile model
  model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.8])   #loss_weights=[0.3]=0.5
# print ("ok2") 
  return model
 
# generator a resnet block
def resnet_block(n_filters, input_layer):
       # print ("ok3") 
  # weight initialization
  init = RandomNormal(stddev=0.02)
  # first layer convolutional layer
  g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # second convolutional layer
  g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  # concatenate merge channel-wise with input layer
  g = Concatenate()([g, input_layer])
# print ("ok4") 
  return g
 
# define the standalone generator model
def define_generator(image_shape, n_resnet=9):
        #print ("ok5") 
  ## weight initialization
  init = RandomNormal(stddev=0.02)
  # image input
  in_image = Input(shape=image_shape)
  # c7s1-64
  g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # d128
  g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # d256
  g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # R256
  for _ in range(n_resnet):
    g = resnet_block(256, g)
  # u128
  g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # u64
  g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  g = Activation('relu')(g)
  # c7s1-3
  g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
  g = InstanceNormalization(axis=-1)(g)
  out_image = Activation('tanh')(g)
  # define model
  model = Model(in_image, out_image)
# print ("ok6") 
  return model
 
# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
     #   print ("ok7") 
  # ensure the model we're updating is trainable
  g_model_1.trainable = True
  # mark discriminator as not trainable
  d_model.trainable = False
  # mark other generator model as not trainable
  g_model_2.trainable = False
  # discriminator element
  input_gen = Input(shape=image_shape)
  gen1_out = g_model_1(input_gen)
  output_d = d_model(gen1_out)
  # identity element
  input_id = Input(shape=image_shape)
  output_id = g_model_1(input_id)
  # forward cycle
  output_f = g_model_2(gen1_out)
  # backward cycle
  gen2_out = g_model_2(input_id)
  output_b = g_model_1(gen2_out)
  # define model graph
  model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
  # define optimization algorithm configuration
  opt = Adam(lr=0.0002, beta_1=0.5)
  # compile model with weighting of least squares loss and L1 loss
  model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
  model.summary()
# print ("ok8") 
  return model
 
# load and prepare training images
def load_real_samples(filename):
       # print ("ok9") 
  # load the dataset
  data = load(filename)
  # unpack arrays
  X1, X2 = data['arr_0'], data['arr_1']
  # scale from [0,255] to [-1,1]
  X1 = (X1 - 127.5) / 127.5
  X2 = (X2 - 127.5) / 127.5
# print ("ok10") 
  return [X1, X2]
 
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
       # print ("ok11") 
  # choose random instances
  ix = randint(0, dataset.shape[0], n_samples)
  # retrieve selected images
  X = dataset[ix]
  # generate 'real' class labels (1)
  y = ones((n_samples, patch_shape, patch_shape, 1))
# print ("ok12") 
  return X, y
 
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
      #  print ("ok13") 
  # generate fake instance
  X = g_model.predict(dataset)
  # create 'fake' class labels (0)
  y = zeros((len(X), patch_shape, patch_shape, 1))
  #print ("ok14") 
  return X, y
 
# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, c_model_AtoB,c_model_BtoA):
       # print ("ok15") 
  # save the first generator model
  filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
  g_model_AtoB.save("/content/drive/My Drive/Deep_learn/photo_eye_tracking/"+str(filename1))
  # save the second generator model
  filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
  print (filename2)
  g_model_BtoA.save("/content/drive/My Drive/Deep_learn/hoto_eye_tracking/"+str(filename2))
  filename3 = 'd_model_A_%06d.h5' % (step+1)
  d_model_A =   d_model_A.save("/content/drive/My Drive/Deep_learn/photo_eye_tracking/"+str(filename3))
  filename4 = 'd_model_B_%06d.h5' % (step+1)
  d_model_B = d_model_B.save("/content/drive/My Drive/photo_eye_tracking/"+str(filename4))
  
  filename5 = 'c_model_AtoB_%06d.h5' % (step+1)
  c_model_AtoB = c_model_AtoB.save("/content/drive/My Drive/Deep_learn/photo_eye_tracking/"+str(filename5))
  filename6 =  'c_model_BtoA_%06d.h5' % (step+1)
  c_model_BtoA = c_model_BtoA.save("/content/drive/My Drive/Deep_learn/photo_eye_tracking/"+str(filename6))
 
 
 
 
# print ("ok16") 
  #print('>Saved: %s and %s' % (filename1, filename2))
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=1):
    #    print ("ok17") 
  # select a sample of input images
  X_in, _ = generate_real_samples(trainX, n_samples, 0)
  # generate translated images
  X_out, _ = generate_fake_samples(g_model, X_in, 0)
  # scale all pixels from [-1,1] to [0,1]
  X_in = (X_in + 1) / 2.0
  X_out = (X_out + 1) / 2.0
  # plot real images
  for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(X_in[i])
  # plot translated image
  for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(X_out[i])
  # save plot to file
  filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
  pyplot.savefig("/content/drive/My Drive/Deep_learn/photo_eye_tracking/"+str(filename1))
  pyplot.close()
    #   print ("ok18") 
# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
  selected = list()
  for image in images:
    if len(pool) < max_size:
      # stock the pool
      pool.append(image)
      selected.append(image)
    elif random() < 0.5:
      # use image, but don't add it to the pool
      selected.append(image)
    else:
      # replace an existing image and use replaced image
      ix = randint(0, len(pool))
      selected.append(pool[ix])
      pool[ix] = image
  return asarray(selected)
 
# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
  # define properties of the training run
  n_epochs, n_batch, = 2, 1 #100
  # determine the output square shape of the discriminator
  n_patch = d_model_A.output_shape[1]
  # unpack dataset
  trainA, trainB = dataset
  # prepare image pool for fakes
  poolA, poolB = list(), list()
  # calculate the number of batches per training epoch
  bat_per_epo = int(len(trainA) / n_batch)
  # calculate the number of training iterations
  n_steps = bat_per_epo * n_epochs
  print ("n_steps",n_steps)
 
  # manually enumerate epochs
  for i in range(n_steps):
    # select a batch of real samples
    X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
    X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
    # generate a batch of fake samples
    X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
    X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
    # update fakes from pool
    X_fakeA = update_image_pool(poolA, X_fakeA)
    X_fakeB = update_image_pool(poolB, X_fakeB)
    # update generator B->A via adversarial and cycle loss
    g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
    # update discriminator for A -> [real/fake]
    dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
    dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
    # update generator A->B via adversarial and cycle loss
    g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
    # update discriminator for B -> [real/fake]
    dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
    dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
    # summarize performance
    print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
    # evaluate the model performance every so often
    if ((i+1) % 50) == 0:   # if (i+1) % (bat_per_epo * 1)
      # plot A->B (translation
      summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
      # plot B->A translation
  # # summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
    print ("for saved",(i+1) % (bat_per_epo * 5))
  # print ("for saved",(i+1))
  # print ("for saved",(bat_per_epo * 5))
 
    if ((i+1) % 350) == 0: #if (i+1) % (bat_per_epo * 5) == 0:
                        
      # save the models
    # save_models(i, g_model_AtoB, g_model_BtoA)
      save_models(i, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B, c_model_AtoB, c_model_BtoA)
      print ("model was saved")
 
# load image data
#from keras_contrib.layers.normalization import InstanceNormalization
 
dataset = load_real_samples('/content/drive/My Drive/Deep_learn/eye_dataset.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
 
#import tensorflow as tf #g_model_AtoB=tf.keras.models.load_model
 
 
 
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# generator: A -> B 
 
#model = keras.models.load_model(self.output_directory + 'best_model.hdf5',
#custom_objects={'InstanceNormalization':InstanceNormalization})
from keras.models import load_model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
 
 

 
#g_model_AtoB.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
# trained_model = load_model('model.h5', compile=False) 
g_model_AtoB.summary()

#g_model_BtoA =load_model('/content/drive/My Drive/Deep_learn/photo/g_model_BtoA_021000.h5',custom_objects={'InstanceNormalization':InstanceNormalization})
#g_model_BtoA.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
g_model_BtoA = define_generator(image_shape)
g_model_BtoA.summary()
# discriminator: A -> [real/fake]
#d_model_A =load_model('/content/drive/My Drive/Deep_learn/photo/d_model_A_021000.h5',custom_objects={'InstanceNormalization':InstanceNormalization})
#d_model_A.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
d_model_A = define_discriminator(image_shape)
d_model_A.summary()
# discriminator: B -> [real/fake]
#d_model_B =load_model('/content/drive/My Drive/Deep_learn/photo/d_model_B_021000.h5',custom_objects={'InstanceNormalization':InstanceNormalization})
#d_model_B.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
d_model_B = define_discriminator(image_shape)
d_model_B.summary()
# composite: A -> B -> [real/fake, A]
#c_model_AtoB  =load_model('/content/drive/My Drive/Deep_learn/photo/c_model_AtoB_021000.h5',custom_objects={'InstanceNormalization':InstanceNormalization})
#c_model_AtoB.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_AtoB.summary()
# composite: B -> A -> [real/fake, B]
#c_model_BtoA =load_model('/content/drive/My Drive/Deep_learn/photo/c_model_BtoA_021000.h5',custom_objects={'InstanceNormalization':InstanceNormalization})
#c_model_BtoA.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
c_model_BtoA.summary()

train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
print ("finish")
