import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

def one_shot_autoencoder_shaper(func):
    def wrapper(vocab_length, *args, **kwargs):
        for gen in func(*args, **kwargs):
            # Padding with -1 to add 1 and have the sequence be padded by 0
            # along with raising the value of all the data by 1
            pad_bat = pad_sequences(gen, padding='post', value=-1) + 1
            hot_out = to_categorical(pad_bat, num_classes=vocab_length + 1)
            yield pad_bat, hot_out
    return wrapper

def forward_autoencoder_shaper(func):
    def wrapper(vocab_length, *args, **kwargs):
        for gen in func(*args, **kwargs):
            # Data is assumed to be indexed from 0 to vocab_length - 1
            # token for Start Of Sequence is indexed as vocab_length
            # token for End of Sequence is indexed as vocab_length + 1
            f_bat = []
            
            sos = vocab_length
            
            for obs in gen:
                f_temp = np.insert(obs, 0, sos)
                # Add 1 to make room for 0 padding
                f_temp = np.delete(f_temp, -1) + 1
                f_bat.append(f_temp)
            
            # Padding with -1 to add 1 and have the sequence be padded by 0
            # along with raising the value of all the data by 1
            pad_bat = pad_sequences(gen, padding='post', value=-1) + 1
            f_bat = pad_sequences(f_bat, padding='post')
            # hot_out will be encoded variables with a length of 
            # vocab_length + 1 to include 0 padding but not Start of Sequence
            # tokens because they will never be predicted.
            hot_out = to_categorical(pad_bat, num_classes=vocab_length + 1)

            yield [pad_bat, f_bat], hot_out
    return wrapper

def bidirectional_autoencoder_shaper(func):
    def wrapper(vocab_length, *args, **kwargs):
        for gen in func(*args, **kwargs):
            # Data is assumed to be indexed from 0 to vocab_length - 1
            # token for Start Of Sequence is indexed as vocab_length
            # token for End of Sequence is indexed as vocab_length + 1
            f_bat = []
            b_bat = []

            sos = vocab_length
            eos = vocab_length + 1
            
            for obs in gen:
                f_temp = np.insert(obs, 0, sos)
                # Add 1 to make room for 0 padding
                f_temp = np.delete(f_temp, -1) + 1
                f_bat.append(f_temp)
                
                b_temp = np.concatenate((obs, [eos]))
                # Add 1 to make room for 0 padding
                b_temp = np.delete(b_temp, 0) + 1
                b_bat.append(b_temp)
            
            # Padding with -1 to add 1 and have the sequence be padded by 0
            # along with raising the value of all the data by 1
            pad_bat = pad_sequences(gen, padding='post', value=-1) + 1
            f_bat = pad_sequences(f_bat, padding='post')
            b_bat = pad_sequences(b_bat, padding='post')
            # hot_out will be encoded variables with a length of 
            # vocab_length + 1 to include 0 padding but not Start of Sequence
            # or End of Sequence tokens because they will never be predicted.
            hot_out = to_categorical(pad_bat, num_classes=vocab_length + 1)

            yield [pad_bat, f_bat, b_bat], hot_out
    return wrapper            

def backward_autoencoder_shaper(func):
    def wrapper(vocab_length, *args, **kwargs):
        for gen in func(*args, **kwargs):
            # Data is assumed to be indexed from 0 to vocab_length - 1
            # token for Start Of Sequence is indexed as vocab_length
            # token for End of Sequence is indexed as vocab_length + 1
            b_bat = []
            
            eos = vocab_length + 1
            
            for obs in gen:
                b_temp = np.concatenate((obs, [eos]))
                # Add 1 to make room for 0 padding
                b_temp = np.delete(b_temp, 0) + 1
                b_bat.append(b_temp)
            
            # Padding with -1 to add 1 and have the sequence be padded by 0
            # along with raising the value of all the data by 1
            pad_bat = pad_sequences(gen, padding='post', value=-1) + 1
            b_bat = pad_sequences(b_bat, padding='post')
            # hot_out will be encoded variables with a length of 
            # vocab_length + 1 to include 0 padding but not End of Sequence
            # tokens because they will never be predicted.
            hot_out = to_categorical(pad_bat, num_classes=vocab_length + 1)
            yield [pad_bat, b_bat], hot_out
    return wrapper      
        
def aux_out_shape(func, aux_num = 1):
    #aux_out_shape_f repeats the output to be put into multiple auxiliary 
    #outputs
    #aux_num is the number of additional auxiliary outputs
    def wrapper(*args, **kwargs):
        for gen in func(*args, **kwargs):
            #gen[0] is assumed to be the inputs, gen[1] is assumed to be the
            #outputs
            new_out = [gen[1] for x in range(aux_num + 1)]
            yield gen[0], new_out
    return wrapper