from keras.layers import BatchNormalization, Input, Embedding, Dense, Bidirectional, Lambda, TimeDistributed, SimpleRNN
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.models import Model
from test_harness import test_harness1
from gen_wrappers import backward_autoencoder_shaper
import numpy as np


def sample_z(z_m_sd):
    z_mean, z_log_var = z_m_sd
    batch_size = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(z_log_var/ 2) * epsilon

def vae_loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        dim = K.int_shape(z_mean)[1]
        xent_loss = dim * binary_crossentropy(y_true, y_pred)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss


latent_dim = 10
vocab_length = 5
emb_dim = 12

inputs = Input(shape=(None,))
embedder = Embedding(vocab_length + 3, emb_dim)
embedded_inputs = embedder(inputs)
h = Bidirectional(SimpleRNN(20), merge_mode='concat')(embedded_inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sample_z, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(inputs,[z_mean, z_log_var, z])

forward_in = Input(shape=(None,))
f_emb = embedder(forward_in)
f_rnn = SimpleRNN(latent_dim, return_sequences=True, return_state=True, go_backwards=True)
h, _ = f_rnn(f_emb, initial_state = z)
h = Lambda(lambda x: K.reverse(x, axes=1))(h)
f_bn = BatchNormalization()
h = f_bn(h)
soft_dense = Dense(vocab_length + 1, activation='softmax')
decoded = TimeDistributed(soft_dense)(h)
#decoded = soft_dense(h)

vae = Model([inputs, forward_in], decoded, name='forward_vae')
vae.compile(optimizer='adam', loss=vae_loss, metrics=['acc'])

test_gen=backward_autoencoder_shaper(test_harness1)(vocab_length = vocab_length)

vae.fit_generator(test_gen, steps_per_epoch=1000, epochs=5)

p = next(test_gen)
prediction = encoder.predict(p[0][0])

current_state = prediction[2]
token = np.zeros((len(current_state), 1))

inference_in = Input(shape=(None,))
in_state = Input(shape=(latent_dim,))
h = embedder(inference_in)
h, out_state = f_rnn(h, initial_state = in_state)
inference_out = soft_dense(h)
decoder = Model([inference_in, in_state], [inference_out, out_state])


current_state = prediction[2]
token = np.zeros((len(current_state), 1))
str_out = token
max_len = 40

while str_out.shape[1] < max_len:
    output, current_state = decoder.predict([np.array(token), current_state])
    token = np.argmax(output, axis = 2)
    str_out = np.concatenate((str_out, token), axis = 1)