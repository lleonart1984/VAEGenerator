import numpy as np
import tensorflow as tf
import math
import random

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

class VAEGenerator:

    def __init__(self, name, parameter_size, sample_size, L):
        self.name = name
        self.rnd = random.Random()
        self.parameter_size = parameter_size
        self.sample_size = sample_size
        self.L = L
        self._buildModels()

    def _buildModels(self):
        # Encoder
        eInputP = tf.keras.Input(shape=(self.parameter_size,), name=self.name+'_encoder_input_parameters')
        eInputX = tf.keras.Input(shape=(self.sample_size,), name=self.name+'_encoder_input_samples')
        mergeInput = tf.keras.layers.concatenate([eInputP, eInputX])
        hLayer = mergeInput
        for _ in range(self.L):
            hLayer = tf.keras.layers.Dense(2*(self.parameter_size+self.sample_size), activation=tf.nn.relu) (hLayer)

        eZMean = tf.keras.layers.Dense(self.sample_size, name='z_mean')(hLayer)
        eZLogVar = tf.keras.layers.Dense(self.sample_size, name='z_log_var')(hLayer)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        eZ = tf.keras.layers.Lambda(sampling, output_shape=(self.sample_size,), name='z')([eZMean, eZLogVar])
        self.encoder = tf.keras.Model([eInputP, eInputX], [eZMean, eZLogVar, eZ], name=self.name+'_encoder')
        self.encoder.summary()

        # Decoder Kernel
        dKInput = tf.keras.Input(shape=(self.parameter_size+self.sample_size,), name=self.name+'_decoderK_input_parameters')
        nodes = int(math.ceil((self.parameter_size + self.sample_size)/4)*4)
        hLayer = dKInput
        for i in range(self.L):
            if i < self.L - 1:
                hLayer = tf.keras.layers.Dense(nodes, activation=tf.nn.relu) (hLayer)
            else:
                hLayer = tf.keras.layers.Dense(nodes, activation=tf.nn.softplus) (hLayer)
        dKOutput = tf.keras.layers.Dense(self.sample_size, activation=tf.nn.softplus) (hLayer)
        self.decoderKernel = tf.keras.Model([dKInput], [dKOutput], name=self.name+'_decoderKernel')

        # Decoder
        dInputP = tf.keras.Input(shape=(self.parameter_size,), name=self.name + '_decoder_input_parameters')
        dInputX = tf.keras.Input(shape=(self.sample_size,), name=self.name + '_decoder_normal_samples')
        mergeInput = tf.keras.layers.concatenate([dInputP, dInputX])
        dOutput = self.decoderKernel(mergeInput)
        self.decoder = tf.keras.Model([dInputP, dInputX], [dOutput], name=self.name+'_decoder')
        self.decoder.summary()

        # Variational Autoencoder
        vaeOutput = self.decoder([eInputP, self.encoder([eInputP, eInputX])[2]])
        self.vae = tf.keras.Model([eInputP, eInputX], [vaeOutput], name=self.name+'_vae')

        #Compiling
        reconstruction_loss = tf.keras.losses.mean_squared_logarithmic_error(eInputX, vaeOutput) * 1000
        kl_loss = 1 + eZLogVar - tf.keras.backend.square(eZMean) - tf.keras.backend.exp(eZLogVar)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        self.vae.summary()
        try:
            self.vae.load_weights(self.name+"_VAEGenerator.tf")
        except:
            pass

    def trainLoop(self, ps, xs, epochs=20):
        # ps : parameters sets
        # xs : list of samples
        parameters = np.array(ps, dtype=float)
        samples = np.array(xs, dtype=float)
        self.vae.fit([parameters, samples], epochs= epochs, batch_size = 128)
    
    def saveModel(self):
        print("saving model...")
        self.vae.save_weights(self.name+"_VAEGenerator.tf")

    def getKernelModel(self):
        return self.decoderKernel

    def generateSamples(self, P, N):
        input = []
        for _ in range(N):
            input.append(P + [self.rnd.gauss(0, 1) for _ in range(self.sample_size)])
        input = np.array(input)
        return self.decoderKernel.predict(input)