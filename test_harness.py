import random
import numpy as np
from keras.utils import to_categorical

def test_harness1(time_steps = 20):
    while 1:
        bat = random.randint(20,50) 
        ran = [random.randint(0,4) for _ in range(bat)]
        x = [[x%(z%5 + 1) for x in range(time_steps)] for z in ran]
        x = np.array(x)
        yield x

def test_harness2(vocab_length, time_steps = 20):
    while 1:
        bat = random.randint(20, 50)   
        x = [[0 for _ in range(time_steps-2)] for y in range(bat)]
        for item in x:
            ran = random.randint(0,4)
            item.insert(0, ran)
            item.append(ran)
        x = np.array(x)
        x_cat = to_categorical(x, num_classes=vocab_length)
        yield (x + 1, x_cat)
        
        