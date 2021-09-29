import os, math
import numpy as np
import tensorflow as tf
import source.utils as utils
import whiteboxlayer.layers as wbl

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.dim_z = kwargs['dim_z']
        self.ksize = kwargs['ksize']
        self.num_flow = kwargs['num_flow']

        filters = kwargs['filters']
        self.filters = [self.dim_c]
        try:
            tmp_filters = filters.split(',')
            for idx, _ in enumerate(tmp_filters):
                tmp_filters[idx] = int(tmp_filters[idx])
            self.filters.extend(tmp_filters)
        except: self.filters.extend([16, 32, 64, 64])

        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.variables = {}

        self.__model = Neuralnet(\
            dim_h=self.dim_h, dim_w=self.dim_w, dim_c=self.dim_c, dim_z=self.dim_z, ksize=self.ksize, num_flow=self.num_flow, filters=self.filters)
        self.__model.forward(\
            x=tf.zeros((1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32), \
            verbose=True)
        print("\nNum Parameter: %d" %(self.__model.layer.num_params))

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(self.path_ckpt)

        self.variables['trainable'] = []
        ftxt = open("list_parameters.txt", "w")
        for key in list(self.__model.layer.parameters.keys()):
            trainable = self.__model.layer.parameters[key].trainable
            text = "T: " + str(key) + str(self.__model.layer.parameters[key].shape)
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.optimizer = tf.optimizers.RMSprop(learning_rate=self.learning_rate)
        self.save_params()

        self.conc_func = self.__model.__call__.get_concrete_function( \
            tf.TensorSpec(shape=(1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32))

    def __loss(self, y, y_hat, ln_q_0, ln_q_k, ln_det):

        restore = self.binary_ce(true=y, pred=y_hat, reduce=(1, 2, 3))

        term_1 = tf.math.reduce_mean(ln_q_0, axis=(1))
        term_2 = -tf.math.reduce_mean(ln_q_k, axis=(1))
        term_3 = -tf.math.reduce_mean(ln_det, axis=(1))
        energy_bound = term_1 + term_2 + term_3

        loss_batch = restore + energy_bound
        loss_mean = tf.math.reduce_mean(loss_batch)

        return {'loss_batch': loss_batch, 'loss_mean': loss_mean, \
            'restore': restore, 'energy_bound': energy_bound, \
            'term_1': term_1, 'term_2': term_2, 'term_3': term_3}

    @tf.autograph.experimental.do_not_convert
    def step(self, minibatch, iteration=0, training=False):

        x, y = minibatch['x'], minibatch['x']

        with tf.GradientTape() as tape:
            outputs = self.__model.forward(x=x, verbose=False)
            y_hat = outputs['y_hat']
            z_0, z_k = outputs['z_0'], outputs['z_k']
            ln_q_0 = outputs['ln_q_0']
            ln_q_k = outputs['ln_q_k']
            ln_det = outputs['ln_det']
            losses = self.__loss(y=y, y_hat=y_hat, ln_q_0=ln_q_0, ln_q_k=ln_q_k, ln_det=ln_det)

        if(training):
            gradients = tape.gradient(losses['loss_mean'], self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('%s/loss_mean' %(self.__model.who_am_i), losses['loss_mean'], step=iteration)

        return {'y_hat':y_hat.numpy(), 'losses':losses, \
            'z_0':z_0.numpy(), 'z_k':z_k.numpy()}

    def save_params(self, model='base'):

        vars_to_save = self.__model.layer.parameters.copy()
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.path_ckpt, model), max_to_keep=1)
        ckptman.save()

    def load_params(self, model):

        vars_to_load = self.__model.layer.parameters.copy()
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(self.path_ckpt, model))
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def binary_ce(self, true, pred, reduce=None):

        return -tf.reduce_mean(true * tf.math.log(pred + 1e-30) + (1 - true) * tf.math.log(1 - pred + 1e-30), axis=reduce)

class Neuralnet(tf.Module):

    def __init__(self, **kwargs):
        super(Neuralnet, self).__init__()

        self.who_am_i = "NF"
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.dim_z = kwargs['dim_z']
        self.ksize = kwargs['ksize']
        self.num_flow = kwargs['num_flow']

        self.filters_enc = kwargs['filters']
        self.filters_dec = self.filters_enc[::-1]

        self.layer = wbl.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, verbose=False):

        latent, list_shape = self.__encoder(x=x, name='enc', verbose=verbose)
        latent = tf.add(latent, 0, name="latent")

        z_0, z_k, ln_q_0, ln_q_k, ln_det = \
            self.__flow(latent=latent, name='enc', verbose=verbose)
        z_k = tf.add(z_k, 0, name="z_k")

        y_hat = self.__decoder(x=z_k, list_shape=list_shape, name='dec', verbose=verbose)
        y_hat = tf.add(y_hat, 0, name="y_hat")

        return {'y_hat':y_hat, 'z_0':z_0, 'z_k':z_k, \
            'ln_q_0':ln_q_0, 'ln_q_k':ln_q_k, 'ln_det':ln_det}

    def __encoder(self, x, name='enc', verbose=True):

        for idx, _ in enumerate(self.filters_enc):
            if(idx == 0): continue
            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[self.ksize, self.ksize, self.filters_enc[idx-1], self.filters_enc[idx]], \
                activation='relu', name='%s-%d_c0' %(name, idx), verbose=verbose)
            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[self.ksize, self.ksize, self.filters_enc[idx], self.filters_enc[idx]], \
                activation='relu', name='%s-%d_c1' %(name, idx), verbose=verbose)
            if(idx < len(self.filters_enc)-1):
                x = self.layer.maxpool(x=x, ksize=2, strides=2, \
                    name='%s-%d_mp' %(name, idx), verbose=verbose)

        [n, h, w, c] = x.shape
        list_shape = [[n, h, w, c]]
        x = tf.compat.v1.reshape(x, shape=[n, h*w*c], name="%s-flat" %(name))
        list_shape.append(x.shape)
        x = self.layer.fully_connected(x=x, c_out=self.dim_z*2, \
            batch_norm=False, activation='relu', name="%s-z0" %(name), verbose=verbose)
        x = tf.clip_by_value(x, -5+(1e-30), 5-(1e-30))

        return x, list_shape

    def __decoder(self, x, list_shape, name='dec', verbose=True):

        x = self.layer.fully_connected(x=x, c_out=list_shape[-1][-1], \
            batch_norm=False, activation='relu', name="%s-z0" %(name), verbose=verbose)
        [n, h, w, c] = list_shape[-2]
        x = tf.compat.v1.reshape(x, shape=[n, h, w, c], name="%s-flat" %(name))

        for idx, _ in enumerate(self.filters_dec):
            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[self.ksize, self.ksize, self.filters_dec[idx], self.filters_dec[idx]], \
                activation='relu', name='%s-%d_c1' %(name, len(self.filters_dec)-(idx+1)), verbose=verbose)
            x = self.layer.conv2d(x=x, stride=1, \
                filter_size=[self.ksize, self.ksize, self.filters_dec[idx], self.filters_dec[idx+1]], \
                activation=None, name='%s-%d_c0' %(name, len(self.filters_dec)-(idx+1)), verbose=verbose)
            if(idx == len(self.filters_dec)-2): break
            x = self.layer.activation(x=x, activation='relu', name='%s-%d_c0_act' %(name, len(self.filters_dec)-(idx+1)))
            [n, h, w, c] = x.shape
            x = self.layer.convt2d(x=x, stride=2, output_shape=[x.shape[0], h*2, w*2, self.filters_dec[idx+1]], \
                filter_size=[self.ksize, self.ksize, self.filters_dec[idx+1], self.filters_dec[idx+1]], dilations=[1, 1, 1, 1], \
                padding='SAME', batch_norm=False, \
                activation='relu', name='%s-%d_c0_up' %(name, len(self.filters_dec)-(idx+1)), verbose=verbose)

        return self.layer.activation(x=x, activation='sigmoid', name='%s-out' %(name))

    def __flow(self, latent, name='flow', verbose=True):

        z1, z2 = tf.split(latent, num_or_size_splits=2, axis=1)
        epsilon = tf.random.normal(tf.shape(z1), dtype=tf.float32)
        z_0 = z1 + (z2 * epsilon)

        w1 = tf.math.sin(((2 * math.pi) * z1) / 4)
        u_z = 0.5 * (z2 - (w1 / 0.4))**2
        ln_q_0 = tf.math.log(tf.math.exp(-u_z))

        z_k, sum_det = z_0, None
        for idx_flow in range(self.num_flow):

            wz = self.layer.fully_connected(x=z_k, c_out=self.dim_z, \
                batch_norm=False, activation=None, name="%s-%d-w" %(name, idx_flow), verbose=verbose)
            h = self.layer.activation(x=wz, activation='tanh', name="%s-%d-wb_act" %(name, idx_flow))
            uh = self.layer.fully_connected(x=h, c_out=self.dim_z, \
                batch_norm=False, activation=None, name="%s-%d-u" %(name, idx_flow), verbose=verbose)
            z_k = z_k + uh
            h_prime = 1 - (wz**2)
            phi_z = self.layer.fully_connected(x=h_prime, c_out=self.dim_z, \
                batch_norm=False, activation=None, name="%s-%d-w" %(name, idx_flow), verbose=verbose)
            uphi_z = self.layer.fully_connected(x=phi_z, c_out=self.dim_z, \
                batch_norm=False, activation=None, name="%s-%d-u" %(name, idx_flow), verbose=verbose)

            det_z_k = tf.math.abs(1 + uphi_z)
            if(sum_det is None): sum_det = tf.math.log(det_z_k + 1e-30)
            else: sum_det += tf.math.log(det_z_k + 1e-30)
        ln_q_k = ln_q_0 - sum_det

        return z_0, z_k, ln_q_0, ln_q_k, sum_det
