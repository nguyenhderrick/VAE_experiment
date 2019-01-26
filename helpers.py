import numpy as np

def hot_zero_tokens(data):
    #creates two rows at the end of the data for the backward and forward
    #start tokens that the inference model will not predict.
    zeros = np.zeros((data.shape[0], data.shape[1], 1))
    data = np.concatenate((data, zeros, zeros),axis=2)
    return data

def guesser_f(generator, data):
    #From the batch normalized layer, guesser takes a sequence of nodes and
    #guesses random values for the second input. 
    noise = np.random.normal(size=np.prod(data.shape))
    noise = np.reshape(noise, data.shape)
    out = generator.predict([data, noise])
    return out

def guesser_b(generator, data):
    #From the batch normalized layer, guesser takes a sequence of nodes and
    #guesses random values for the second input. 
    noise = np.random.normal(size=np.prod(data.shape))
    noise = np.reshape(noise, data.shape)
    out = generator.predict([noise, data])
    return out

def guesser_zero_f(generator, data, class_prob=True):
    #From the batch normalized layer, guesser takes a sequence of nodes and
    #guesses zero for the second input. Then it predicts the class
    #based on guesses and takes the value most guessed in the number of 
    #iterations.
    zero = np.zeros(data.shape)
    out = generator.predict([data, zero])
    if class_prob is False:
        out = np.argmax(out[:,-1,:],axis=1)
        out = np.reshape(out, (-1,1))
    return out

def guesser_zero_b(generator, data, class_prob=True):
    #From the batch normalized layer, guesser takes a sequence of nodes and
    #guesses zero for the second input. Then it predicts the class
    #based on guesses and takes the value most guessed in the number of 
    #iterations.
    zero = np.zeros(data.shape)
    out = generator.predict([zero, data])
    if class_prob is False:
        out = np.argmax(out[:,-1,:],axis=1)
        out = np.reshape(out, (-1,1))
    return out