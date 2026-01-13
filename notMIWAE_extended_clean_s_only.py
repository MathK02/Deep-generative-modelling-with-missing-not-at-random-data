import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
import keras
import numpy as np
import datetime


class notMIWAE_extended_2:
    """
    Extended not-MIWAE with regime encoder q(u|s) ONLY (not using x)
    """
    
    # we have a new parameter to control the dimension of the latent variable, n_u_transform

    def __init__(self, X, Xval,
                 n_latent=50, n_latent_u=10, n_hidden=100, n_u_transform=15, n_samples=1,
                 activation=tf.nn.tanh,
                 out_dist='gauss',
                 out_activation=None,
                 learnable_imputation=False,
                 permutation_invariance=False,
                 embedding_size=20,
                 code_size=20,
                 missing_process='linear',
                 missing_model_architecture='concat',
                 testing=False,
                 name='/tmp/notMIWAE'):

        # ---- data
        self.Xorg = X.copy()
        self.Xval_org = Xval.copy()
        self.n, self.d = X.shape

        # ---- missing
        self.S = np.array(~np.isnan(X), dtype=np.float32)
        self.Sval = np.array(~np.isnan(Xval), dtype=np.float32)

        if np.sum(self.S) < self.d * self.n:
            self.X = self.Xorg.copy()
            self.X[np.isnan(self.X)] = 0
            self.Xval = self.Xval_org.copy()
            self.Xval[np.isnan(self.Xval)] = 0
        else:
            self.X = self.Xorg
            self.Xval = self.Xval_org

        # ---- settings
        self.n_latent = n_latent
        self.n_latent_u = n_latent_u
        self.n_hidden = n_hidden
        self.n_u_transform = n_u_transform
        self.n_samples = n_samples
        self.activation = activation
        self.out_dist = out_dist
        self.out_activation = out_activation
        self.embedding_size = embedding_size
        self.code_size = code_size
        self.missing_process = missing_process
        self.missing_model_architecture = missing_model_architecture
        self.testing = testing
        self.batch_pointer = 0
        self.eps = np.finfo(float).eps

        print("Creating graph...")
        tf.reset_default_graph()

        # ---- input
        with tf.variable_scope('input'):
            self.x_pl = tf.placeholder(tf.float32, [None, self.d], 'x_pl')
            self.s_pl = tf.placeholder(tf.float32, [None, self.d], 's_pl')
            self.n_pl = tf.placeholder(tf.int32, shape=(), name='n_pl')

        if learnable_imputation and not testing:
            self.imp = tf.get_variable('imp', shape=[1, self.d])
            self.in_pl = self.x_pl + (1 - self.s_pl) * self.imp
        elif permutation_invariance and not testing:
            self.in_pl = self.permutation_invariant_embedding()
        else:
            self.in_pl = self.x_pl

        # ---- parameters from encoder
        with tf.variable_scope('encoder'):
            self.q_mu, self.q_log_sig2 = self.encoder(self.in_pl)

        # ---- parameters from regime encoder (u from s ONLY)
        with tf.variable_scope('regime_encoder'):
            self.q_mu_u, self.q_log_sig2_u = self.regime_encoder(self.s_pl)

        # ---- variational distribution
        q_z = tfp.distributions.Normal(loc=self.q_mu, scale=tf.sqrt(tf.exp(self.q_log_sig2)))
        q_u = tfp.distributions.Normal(loc=self.q_mu_u, scale=tf.sqrt(tf.exp(self.q_log_sig2_u)))

        # ---- sample the latent values
        self.l_z = q_z.sample(self.n_pl)
        self.l_z = tf.transpose(self.l_z, perm=[1, 0, 2])
        
        self.l_u = q_u.sample(self.n_pl)
        self.l_u = tf.transpose(self.l_u, perm=[1, 0, 2])

        # ---- parameters from decoder
        if out_dist in ['gauss', 'normal', 'truncated_normal']:

            with tf.variable_scope('data_process'):
                mu, std = self.gauss_decoder(self.l_z)

            # ---- p(x|z)
            if out_dist == 'truncated_normal':
                p_x_given_z = tfp.distributions.TruncatedNormal(loc=mu, scale=std, low=0.0, high=1.0)
            else:
                p_x_given_z = tfp.distributions.Normal(loc=mu, scale=std)

            # ---- evaluate x in p(x|z)
            self.log_p_x_given_z = tf.reduce_sum(
                tf.expand_dims(self.s_pl, axis=1) * p_x_given_z.log_prob(tf.expand_dims(self.x_pl, axis=1)), axis=-1)

            self.l_out_mu = mu
            # ---- sample xm from p(x|z)
            self.l_out_sample = p_x_given_z.sample()

        elif out_dist == 'bern':

            with tf.variable_scope('data_process'):
                logits = self.bernoulli_decoder(self.l_z)

            # ---- p(x|z)
            p_x_given_z = tfp.distributions.Bernoulli(logits=logits)

            self.log_p_x_given_z = tf.reduce_sum(
                tf.expand_dims(self.s_pl, axis=1) * p_x_given_z.log_prob(tf.expand_dims(self.x_pl, axis=1)), axis=-1)

            self.l_out_mu = tf.nn.sigmoid(logits)
            # ---- sample xm from p(x|z)
            self.l_out_sample = tf.cast(p_x_given_z.sample(), tf.float32)

        elif out_dist in ['t', 't-distribution']:

            with tf.variable_scope('decoder'):
                mu, log_sig2, df = self.t_decoder(self.l_z)

            # ---- p(x|z)
            p_x_given_z = tfp.distributions.StudentT(loc=mu,
                                                     scale=tf.nn.softplus(log_sig2) + 0.0001,
                                                     df=3 + tf.nn.softplus(df))

            self.log_p_x_given_z = tf.reduce_sum(
                tf.expand_dims(self.s_pl, axis=1) * p_x_given_z.log_prob(tf.expand_dims(self.x_pl, axis=1)), axis=-1)

            self.l_out_mu = mu
            self.l_out_sample = p_x_given_z.sample()

        else:
            print("use 'gauss', 'normal', 'truncated_normal' or 'bern' as out_dist")

        # ---- the missing process
        with tf.variable_scope('missing'):

            # ---- mix x_o with samples of x_m
            self.l_out_mixed = self.l_out_sample * tf.expand_dims(1 - self.s_pl, axis=1) + tf.expand_dims(
                self.x_pl * self.s_pl, axis=1)

            self.logits_miss = self.bernoulli_decoder_miss(self.l_out_mixed, self.l_u)

        # ---- p(s|x,u)
        self.p_s_given_x_u = tfp.distributions.Bernoulli(logits=self.logits_miss)
        
        # ---- evaluate s in p(s|x,u)
        self.log_p_s_given_x_u = tf.reduce_sum(self.p_s_given_x_u.log_prob(tf.expand_dims(self.s_pl, axis=1)), axis=-1)
        
        # ---- alias for compatibility
        self.log_p_s_given_x = self.log_p_s_given_x_u

        # --- evaluate the z-samples in q(z|x)
        q_z2 = tfp.distributions.Normal(loc=tf.expand_dims(self.q_mu, axis=1),
                                       scale=tf.sqrt(tf.exp(tf.expand_dims(self.q_log_sig2, axis=1))))
        self.log_q_z_given_x = tf.reduce_sum(q_z2.log_prob(self.l_z), axis=-1)

        # --- evaluate the u-samples in q(u|s)
        q_u2 = tfp.distributions.Normal(loc=tf.expand_dims(self.q_mu_u, axis=1),
                                       scale=tf.sqrt(tf.exp(tf.expand_dims(self.q_log_sig2_u, axis=1))))
        self.log_q_u_given_s = tf.reduce_sum(q_u2.log_prob(self.l_u), axis=-1)

        # ---- evaluate the z-samples in the prior
        prior_z = tfp.distributions.Normal(loc=0.0, scale=1.0)
        self.log_p_z = tf.reduce_sum(prior_z.log_prob(self.l_z), axis=-1)

        # ---- evaluate the u-samples in the prior
        prior_u = tfp.distributions.Normal(loc=0.0, scale=1.0)
        self.log_p_u = tf.reduce_sum(prior_u.log_prob(self.l_u), axis=-1)

        # ---- notMIWAE extended
        self.notMIWAE = self.get_notMIWAE_extended(self.log_p_x_given_z,
                                                    self.log_p_s_given_x_u,
                                                    self.log_q_z_given_x,
                                                    self.log_q_u_given_s,
                                                    self.log_p_z,
                                                    self.log_p_u)
        # ---- MIWAE for test-set LLH
        self.MIWAE = self.get_MIWAE(self.log_p_x_given_z,
                                    self.log_q_z_given_x,
                                    self.log_p_z)

        # ---- loss
        if self.testing:
            self.loss = - self.MIWAE
        else:
            self.loss = - self.notMIWAE

        # ---- training stuff
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        self.optimizer = tf.train.AdamOptimizer()
        if self.testing:
            tvars = tf.trainable_variables(scope='encoder')
        else:
            tvars = tf.trainable_variables()
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step, var_list=tvars)

        self.sess.run(tf.global_variables_initializer())

        if permutation_invariance:
            svars = tf.trainable_variables('data_process')
            svars.append(self.global_step)
            self.saver = tf.train.Saver(svars)
        else:
            self.saver = tf.train.Saver()

        tf.summary.scalar('Evaluation/loss', self.loss)
        tf.summary.scalar('Evaluation/pxz', tf.reduce_mean(self.log_p_x_given_z))
        tf.summary.scalar('Evaluation/psxu', tf.reduce_mean(self.log_p_s_given_x_u))
        tf.summary.scalar('Evaluation/qzx', tf.reduce_mean(self.log_q_z_given_x))
        tf.summary.scalar('Evaluation/qus', tf.reduce_mean(self.log_q_u_given_s))
        tf.summary.scalar('Evaluation/pz', tf.reduce_mean(self.log_p_z))
        tf.summary.scalar('Evaluation/pu', tf.reduce_mean(self.log_p_u))

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.FileWriter(name + '/tensorboard/notmiwae_train/{}/'.format(timestamp),
                                                  self.sess.graph)
        self.val_writer = tf.summary.FileWriter(name + '/tensorboard/notmiwae_val/{}/'.format(timestamp),
                                                self.sess.graph)
        self.summaries = tf.summary.merge_all()

    def encoder(self, x):

        x = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_enc1')(x)
        x = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_enc2')(x)

        mu = keras.layers.Dense(units=self.n_latent, activation=None, name='q_mu')(x)

        log_sig2 = keras.layers.Dense(units=self.n_latent, activation=lambda x: tf.clip_by_value(x, -10, 10),
                                   name='q_log_sigma')(x)

        return mu, log_sig2

    #This is the encoder of u. In this verion, we take into account only s (no x_observed as in 1st version)
    def regime_encoder(self, s):
        """Regime encoder q(u|s) - encodes ONLY from mask"""
        
        h = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_regime_s')(s)

        mu_u = keras.layers.Dense(units=self.n_latent_u, activation=None, name='q_mu_u')(h)

        log_sig2_u = keras.layers.Dense(units=self.n_latent_u, activation=lambda x: tf.clip_by_value(x, -10, 10),
                                   name='q_log_sigma_u')(h)

        return mu_u, log_sig2_u

    def gauss_decoder(self, z):

        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_dec_gauss1')(z)
        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_dec_gauss2')(z)

        mu = keras.layers.Dense(units=self.d, activation=self.out_activation, name='mu')(z)

        std = keras.layers.Dense(units=self.d, activation=tf.nn.softplus, name='std')(z)

        return mu, std

    def bernoulli_decoder(self, z):

        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_dec_bern1')(z)
        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, name='l_dec_bern2')(z)

        logits = keras.layers.Dense(units=self.d, activation=None, name='logits')(z)

        return logits

    def t_decoder(self, z):

        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, kernel_initializer='orthogonal', name='l_dec1')(z)
        z = keras.layers.Dense(units=self.n_hidden, activation=self.activation, kernel_initializer='orthogonal', name='l_dec2')(z)

        mu = keras.layers.Dense(units=self.d, activation=self.out_activation, kernel_initializer='orthogonal', name='mu')(z)

        log_sigma = keras.layers.Dense(units=self.d, activation=lambda x: tf.clip_by_value(x, -10, 10),
                                       kernel_initializer='orthogonal',
                                       name='log_sigma')(z)

        df = keras.layers.Dense(units=self.d, activation=None, kernel_initializer='orthogonal', name='df')(z)

        return mu, log_sigma, df


    #We change the decoder of the mask to make it take into account both u and x, we can take u into account with 3 distincts ways
        #We can concatenate x and u in various ways:
            #-linear is just the simple concatenation
            #- non linear is the concatenation with tanh activation (2 layers instead of 1)
            #-hybrid: we apply a layer to u before concatenation with x.
        
        #We can multiply u transformed by a layer with x with "multiplicative"
        #We can sum u and x which is very similar to linear concat except from the bias that is not gonna be common between x and u
    
    def bernoulli_decoder_miss(self, x, u):


        if self.missing_model_architecture == 'concat':
            
            if self.missing_process == 'linear':
                concat = tf.concat([x, u], axis=-1)
                logits = keras.layers.Dense(units=self.d, activation=None, name='miss_linear')(concat)
                
            elif self.missing_process == 'nonlinear':
                concat = tf.concat([x, u], axis=-1)
                h = keras.layers.Dense(units=self.n_hidden, activation=tf.nn.tanh, name='miss_hidden')(concat)
                logits = keras.layers.Dense(units=self.d, activation=None, name='miss_out')(h)
            
            elif self.missing_process == 'hybrid':
                u_transform = keras.layers.Dense(units=self.n_u_transform, activation=tf.nn.tanh, name='u_transform')(u)
                concat = tf.concat([x, u_transform], axis=-1)
                logits = keras.layers.Dense(units=self.d, activation=None, name='miss_hybrid')(concat)
                
            else:
                print(f"use 'linear', 'nonlinear' or 'hybrid' as 'missing_process' for concat architecture")
                concat = tf.concat([x, u], axis=-1)
                logits = keras.layers.Dense(units=self.d, activation=None, name='miss_default')(concat)
        
        elif self.missing_model_architecture == 'additive':
            logits_x = keras.layers.Dense(units=self.d, activation=None, name='miss_x')(x)
            logits_u = keras.layers.Dense(units=self.d, activation=None, name='miss_u')(u)
            logits = logits_x + logits_u
        
        elif self.missing_model_architecture == 'multiplicative':
            u_transform = keras.layers.Dense(units=self.d, activation=tf.nn.tanh, name='miss_u_transform')(u)
            x_weighted = x * u_transform
            logits = keras.layers.Dense(units=self.d, activation=None, name='miss_out')(x_weighted)
        
        else:
            print(f"use 'concat', 'additive' or 'multiplicative' as 'missing_model_architecture'")
            concat = tf.concat([x, u], axis=-1)
            logits = keras.layers.Dense(units=self.d, activation=None, name='miss_fallback')(concat)

        return logits

    #The loss changes because of the introduction of u as explained in the report
    def get_notMIWAE_extended(self, lpxz, lpsxu, lqzx, lqus, lpz, lpu):

        # ---- importance weights
        l_w = lpxz + lpsxu + lpz + lpu - lqzx - lqus

        # ---- sum over samples
        log_sum_w = tf.reduce_logsumexp(l_w, axis=1)

        # ---- average over samples
        log_avg_weight = log_sum_w - tf.log(tf.cast(self.n_pl, tf.float32))

        # ---- average over minibatch
        return tf.reduce_mean(log_avg_weight, axis=-1)

    def get_MIWAE(self, lpxz, lqzx, lpz):

        # ---- importance weights
        l_w = lpxz + lpz - lqzx

        # ---- sum over samples
        log_sum_w = tf.reduce_logsumexp(l_w, axis=1)

        # ---- average over samples
        log_avg_weight = log_sum_w - tf.log(tf.cast(self.n_pl, tf.float32))

        # ---- average over minibatch
        return tf.reduce_mean(log_avg_weight, axis=-1)

    def permutation_invariant_embedding(self):

        self.E = tf.get_variable('E', shape=[self.d, self.embedding_size])

        self.Es = tf.expand_dims(self.s_pl, axis=2) * tf.expand_dims(self.E, axis=0)
        print("Es", self.Es.shape)

        self.Esx = tf.concat([self.Es, tf.expand_dims(self.x_pl, axis=2)], axis=2)
        print("Esx", self.Esx.shape)

        self.Esxr = tf.reshape(self.Esx, [-1, self.embedding_size + 1])
        print("Esxr", self.Esxr.shape)

        self.h = keras.layers.Dense(units=self.code_size, activation=tf.nn.relu, name='h1')(self.Esxr)
        print("h", self.h.shape)

        self.hr = tf.reshape(self.h, [-1, self.d, self.code_size])
        print("hr", self.hr.shape)

        self.hz = tf.expand_dims(self.s_pl, axis=2) * self.hr
        print("hz", self.hz.shape)

        self.g = tf.reduce_sum(self.hz, axis=1)
        print("g", self.g.shape)

        return self.g

    def train_batch(self, batch_size):

        x_batch = self.X[self.batch_pointer: self.batch_pointer + batch_size, :]
        s_batch = self.S[self.batch_pointer: self.batch_pointer + batch_size, :]

        _, _loss, _step = \
            self.sess.run([self.train_op, self.loss, self.global_step],
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

        self.tick_batch_pointer(batch_size)

        return _loss

    def val_batch(self):

        batch_size = 100
        val_loss = 0.0
        pxz = 0.0
        psxu = 0.0
        pz = 0.0
        pu = 0.0
        qzx = 0.0
        qus = 0.0
        n_val_batches = len(self.Xval) // batch_size

        for i in range(n_val_batches):

            x_batch = self.Xval[i * batch_size: (i + 1) * batch_size]
            s_batch = self.Sval[i * batch_size: (i + 1) * batch_size]

            _loss, _pxz, _psxu, _qzx, _qus, _pz, _pu, _step = \
                self.sess.run([self.loss, self.log_p_x_given_z, self.log_p_s_given_x_u, 
                              self.log_q_z_given_x, self.log_q_u_given_s, 
                              self.log_p_z, self.log_p_u, self.global_step],
                              {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

            val_loss += _loss
            pxz += np.mean(_pxz)
            psxu += np.mean(_psxu)
            pz += np.mean(_pz)
            pu += np.mean(_pu)
            qzx += np.mean(_qzx)
            qus += np.mean(_qus)

        val_loss /= n_val_batches
        pxz /= n_val_batches
        psxu /= n_val_batches
        pz /= n_val_batches
        pu /= n_val_batches
        qzx /= n_val_batches
        qus /= n_val_batches

        summary = tf.Summary()
        summary.value.add(tag="Evaluation/loss", simple_value=val_loss)
        summary.value.add(tag="Evaluation/pxz", simple_value=pxz)
        summary.value.add(tag="Evaluation/psxu", simple_value=psxu)
        summary.value.add(tag="Evaluation/qzx", simple_value=qzx)
        summary.value.add(tag="Evaluation/qus", simple_value=qus)
        summary.value.add(tag="Evaluation/pz", simple_value=pz)
        summary.value.add(tag="Evaluation/pu", simple_value=pu)

        self.val_writer.add_summary(summary, _step)
        self.val_writer.flush()

        x_batch = self.X[self.batch_pointer: self.batch_pointer + batch_size, :]
        s_batch = self.S[self.batch_pointer: self.batch_pointer + batch_size, :]

        _step, _summaries= \
            self.sess.run([self.global_step, self.summaries],
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: self.n_samples})

        self.train_writer.add_summary(_summaries, _step)
        self.train_writer.flush()

        return val_loss

    def get_llh_estimate(self, Xtest, n_samples=100):
        x_batch = Xtest
        s_batch = (~np.isnan(Xtest)).astype(np.float32)

        _llh = self.sess.run(self.MIWAE,
                          {self.x_pl: x_batch, self.s_pl: s_batch, self.n_pl: n_samples})

        return _llh

    def tick_batch_pointer(self, batch_size):
        self.batch_pointer += batch_size
        if self.batch_pointer >= self.n - batch_size:
            self.batch_pointer = 0

            try:
                p = np.random.permutation(self.n)
                self.X = self.X[p, :]
                self.S = self.S[p, :]
            except MemoryError as error:
                print("Memory error: no shuffling this time")
                print(error)
            except Exception as exception:
                print("Unexpected exception")
                print(exception)

    def save(self, name):
        print("Saving session...")
        self.saver.save(self.sess, name)

    def load(self, name):
        print("Restoring session...")
        self.saver.restore(self.sess, name)
        print("Session restored from global step ", self.sess.run(self.global_step))