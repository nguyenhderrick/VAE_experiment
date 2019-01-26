from gen_wrappers import bidirectional_autoencoder_shaper, aux_out_shape
from test_harness import test_harness1
from VAE_model import encoder, embedder, inference_f, inference_b, decoder_2, \
vae, vocab_length, emb_dim
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from helpers import guesser_zero_f, hot_zero_tokens

test_gen = aux_out_shape(bidirectional_autoencoder_shaper(test_harness1), 2)(vocab_length)
vae.fit_generator(test_gen, steps_per_epoch=600, epochs=5)

p = next(test_gen)
prediction = encoder.predict(p[0][0])

token_f = np.tile(np.array(vocab_length + 1), (prediction[2].shape[0],1,1))
token_f = to_categorical(token_f, vocab_length + 3)

token_b = np.tile(np.array(vocab_length + 2), (prediction[2].shape[0],1,1))
token_b = to_categorical(token_b, vocab_length + 3)

in_z_data = prediction[2]

fake_input = Input(shape=(None, vocab_length + 3))
fake_emb = Dense(emb_dim, use_bias=False, weights=embedder.get_weights())
fake_out = fake_emb(fake_input)
fake_embedder = Model(fake_input, fake_out)

in_f_data = token_f  
while in_f_data.shape[1] < p[0][0].shape[1]:
    emb_f_data = fake_embedder.predict(in_f_data)
    out_f = inference_f.predict([emb_f_data, in_z_data])
    guess_f = guesser_zero_f(decoder_2, out_f)
    guess_f = hot_zero_tokens(guess_f)
    in_f_data = np.concatenate((token_f, guess_f), axis=1)
  
def add_noise(array, scale=0.2):
    noise = np.random.normal(scale=scale, size=array.shape)
    return np.add(array, noise)

reps = 10
repsf = 2
repsr = 2

for r in range(reps):
    rf = 0
    rep = 0
    while rf < repsf and rep <= 10:
        emb_f_data = fake_embedder.predict(in_f_data)
        out_f = inference_f.predict([emb_f_data, in_z_data])
        
        in_b_data = token_b
        for t in reversed(range(in_f_data.shape[1]-1)):
            emb_b_data = fake_embedder.predict(in_b_data)
            out_b = inference_b.predict([add_noise(emb_b_data), in_z_data])
            next_out = decoder_2.predict([out_f[:,[t],:], out_b[:,[0],:]])
            in_b_data = np.concatenate((hot_zero_tokens(next_out), in_b_data), axis=1)
        
        in_f_data2 = np.concatenate((token_f, in_b_data[:,:-1,:]), axis=1)
        if np.array_equal(np.argmax(in_f_data, axis=2), np.argmax(in_f_data2, axis=2)):
            rf += 1
        else:
            rf = 0
        rep += 1
        in_f_data = in_f_data2
    
    rr = 0
    rep = 0
    while rr < repsr and rep <= 10:
        emb_b_data = fake_embedder.predict(in_b_data)
        out_b = inference_b.predict([emb_b_data, in_z_data])

        
        in_f_data = token_f    
        for t in reversed(range(in_b_data.shape[1]-1)):
            emb_f_data = fake_embedder.predict(in_f_data)
            out_f = inference_f.predict([add_noise(emb_f_data), in_z_data])
            next_out = decoder_2.predict([out_f[:,[-1],:], out_b[:,[t],:]])
            in_f_data = np.concatenate((in_f_data,hot_zero_tokens(next_out)), axis=1)

        in_b_data2 = np.concatenate((in_f_data[:,1:,:], token_b), axis=1)     
        if np.array_equal(np.argmax(in_b_data, axis=2), np.argmax(in_b_data2, axis=2)):
            rr += 1
        else:
            rr = 0
        rep += 1
        in_b_data = in_b_data2
    
        
        
print(np.argmax(in_f_data, axis=2))
print(np.amax(p[0][0], axis=1))