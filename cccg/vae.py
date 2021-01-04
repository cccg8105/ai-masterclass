# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:39:53 2021

@author: César Cárdenas
"""

# Building the CNN-VAE model
 
# Importing the libraries
 
import numpy as np
import tensorflow as tf
 
# Building the CNN-VAE model within a class
 
class ConvVAE(object):
    
    # Inicializacion de parametros de clase ConvVAE 
    # z_size: tamaño del vector latente
    # batch_size: tamaño de lote de entrenamiento
    # learning_rate: ratio de aprendizaje
    # kl_tolerance: Parametro de la funcion de perdidas
    # is_training: Define si se esta entrenando o infiriendo
    # reuse: Define si el ambito de visibilidad de variables se reutilizan
    # gpu_mode: Define si se usa la gpu para el entrenamiento 
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse
        # Se crea ambito de visibilidad para variables
        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if not gpu_mode:
                # indica que se entrene con la primera cpu disponible
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self._build_graph()
            else:
                tf.logging.info('Model using gpu.')
                self._build_graph()
        self._init_session()
    
    # Construcción de modelo VAE
    def _build_graph(self):
        # se crea grafo de tensorflow
        self.g = tf.Graph()
        # se construye el grafo
        with self.g.as_default():
            # Se especifica el tamaño de la imagen de entrada
            self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
            # Construcción de capas de convolucion
            h = tf.layers.conv2d(self.x, filters=32,kernel_size= 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
            h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
            h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
            h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
            # aplanado de la salida de la ultima capa
            h = tf.reshape(h, [-1, 2*2*256])
            # Parametros de modelo variacional
            self.mu = tf.layers.dense(h, units = self.z_size, name="enc_fc_mu")
            self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
            self.sigma = tf.exp(self.logvar / 2.0)
            self.epsilon = tf.random_normal([self.batch_size, self.z_size])
            self.z = self.mu + self.sigma * self.epsilon
            # Proceso de deconvolucion
            h = tf.layers.dense(self.z, 1024, name="dec_fc")
            h = tf.reshape(h, [-1, 1, 1, 1024])
            h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
            h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
            h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
            self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")
            # Entrenamiento con gradiente decedente estocastico
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.r_loss = tf.reduce_sum(tf.square(self.x - self.y), reduction_indices = [1,2,3])
                self.r_loss = tf.reduce_mean(self.r_loss)
                self.kl_loss = - 0.5 * tf.reduce_sum((1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indices = 1)
                self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
                self.kl_loss = tf.reduce_mean(self.kl_loss)
                self.loss = self.r_loss + self.kl_loss
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                grads = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')
            self.init = tf.global_variables_initializer()