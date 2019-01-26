from keras.models import Model
from keras.layers import Input, Embedding, Dense, Bidirectional, \
Lambda, RepeatVector, SimpleRNN, Concatenate, BatchNormalization, GaussianNoise
from keras.losses import binary_crossentropy
from keras import backend as K
from keras import regularizers

latent_dim = 10
vocab_length = 5
rnn_dim = 10
emb_dim = 5

def sample_z(z_m_sd):
    z_mean, z_log_var = z_m_sd
    batch_size = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(z_log_var/ 2) * epsilon

def repeat_vector(args):
    z = args[0]
    seq_var = args[1]
    return RepeatVector(K.shape(seq_var)[1])(z)

def vae_loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        dim = K.int_shape(z_mean)[1]
        xent_loss = dim * binary_crossentropy(y_true, y_pred)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

inputs = Input(shape=(None,))
# vocab_length + 3 represents 0 padding, start token, and end token
embedder = Embedding(vocab_length + 3, emb_dim)
embedded_inputs = embedder(inputs)
h = Bidirectional(SimpleRNN(20), merge_mode='concat')(embedded_inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sample_z, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(inputs,[z_mean, z_log_var, z])

repeat_cust = Lambda(repeat_vector, output_shape=(None, latent_dim))
z_in = Input(shape=(latent_dim,))

forward_in = Input(shape=(None,))
f_emb = embedder(forward_in)
forward_embedder = Model(forward_in, f_emb)

f_emb_in = Input(shape=f_emb[0].shape.as_list())
f_emb_in_noisy = GaussianNoise(1)(f_emb_in)
f_z = repeat_cust([z_in, f_emb_in])
f_z_emb = Concatenate()([f_z, f_emb_in_noisy])
f_rnn = SimpleRNN(rnn_dim, 
                  recurrent_regularizer=regularizers.l2(0.3),
                  return_sequences=True, 
                  return_state=True)
f_h, _ = f_rnn(f_z_emb)
f_bn = BatchNormalization()
f_h = f_bn(f_h)
inference_f = Model([f_emb_in, z_in], f_h)

backward_in = Input(shape=(None,))
b_emb = embedder(backward_in)
backward_embedder = Model(backward_in, b_emb)

b_emb_in = Input(shape=b_emb[0].shape.as_list())
b_emb_in_noisy = GaussianNoise(1)(b_emb_in)
b_z = repeat_cust([z_in, b_emb_in])
b_z_emb = Concatenate()([b_z, b_emb_in_noisy])
b_rnn = SimpleRNN(rnn_dim, 
                  recurrent_regularizer=regularizers.l2(0.3),
                  return_sequences=True, 
                  return_state=True, 
                  go_backwards=True)
b_h, _ = b_rnn(b_z_emb)
b_h = Lambda(lambda x: K.reverse(x, axes=1))(b_h)
b_bn = BatchNormalization()
b_h = b_bn(b_h)
inference_b = Model([b_emb_in, z_in], b_h)

decoder_1 = Model([f_emb_in, b_emb_in, z_in], [f_h, b_h])

f_bn_in = Input(shape=f_h[0].shape.as_list())
b_bn_in = Input(shape=b_h[0].shape.as_list())
h = Concatenate()([f_bn_in, b_bn_in])
soft_dense = Dense(vocab_length + 1, activation='softmax')
output = soft_dense(h)

decoder_2 = Model([f_bn_in, b_bn_in], output)

f_soft_dense = Dense(vocab_length + 1, activation='softmax')
b_soft_dense = Dense(vocab_length + 1, activation='softmax')
f_out = f_soft_dense(f_bn_in)
b_out = b_soft_dense(b_bn_in)
aux_decoder_f = Model(f_bn_in, f_out)
aux_decoder_b = Model(b_bn_in, b_out)

f_emb_out = forward_embedder(forward_in)
b_emb_out  = backward_embedder(backward_in)

output_layer = decoder_2(decoder_1([f_emb_out, 
                                    b_emb_out, 
                                    encoder(inputs)[2]]))

aux_out_f = aux_decoder_f(inference_f([f_emb_out, encoder(inputs)[2]]))
aux_out_b = aux_decoder_b(inference_b([b_emb_out, encoder(inputs)[2]]))

vae = Model([inputs, forward_in, backward_in], [output_layer, aux_out_f, aux_out_b], name='bidirectional_vae')
vae.compile(optimizer='adam', loss=vae_loss, metrics=['acc'])


