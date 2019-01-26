import pickle
import numpy as np
from bucket_gen import bucket_gen

vocab_index = pickle.load(open("vocabIndex.p", "rb"))
vocab_dict = pickle.load(open("vocabDict.p", "rb"))

max_length = 150
vocab_length = sum([len(values) for key, values in vocab_dict.items()])
latent_dim = 10

vocab_length = 5

index_string = []
for df in vocab_index:
    index_string.append(df.values.flatten())
    
index_length = [len(df) for df in vocab_index]
include_index = [item < max_length for item in index_length]
index_string = np.array(index_string)[np.array(include_index)]

chord_batch = bucket_gen(vocab_length, index_string, 300, 100, padding='post')