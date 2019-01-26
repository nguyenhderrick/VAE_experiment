from gen_wrappers import bidirectional_autoencoder_shaper, aux_out_shape
from test_harness import test_harness1
from VAE_model import encoder, embedder, inference_f, vae, vocab_length, \
emb_dim, aux_decoder_f
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from helpers import hot_zero_tokens

test_gen = aux_out_shape(bidirectional_autoencoder_shaper(test_harness1), 2)(vocab_length)
vae.fit_generator(test_gen, steps_per_epoch=600, epochs=5)

p = next(test_gen)
prediction = encoder.predict(p[0][0])

token_f = np.tile(np.array(vocab_length + 1), (prediction[2].shape[0],1,1))
token_f = to_categorical(token_f, vocab_length + 3)

in_z_data = prediction[2]

fake_input = Input(shape=(None, vocab_length + 3))
fake_emb = Dense(emb_dim, use_bias=False, weights=embedder.get_weights())
fake_out = fake_emb(fake_input)
fake_embedder = Model(fake_input, fake_out)

in_f_data = token_f      
while in_f_data.shape[1] < p[0][0].shape[1]:
    emb_f_data = fake_embedder.predict(in_f_data)
    out_f = inference_f.predict([emb_f_data, in_z_data])
    guess_f = aux_decoder_f.predict(out_f)
    guess_f = hot_zero_tokens(guess_f)
    in_f_data = np.concatenate((token_f, guess_f), axis=1)

print(np.argmax(in_f_data, axis=2))
print(np.amax(p[0][0], axis=1))