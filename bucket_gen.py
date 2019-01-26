import random
import numpy as np

def bucket_gen(data, batch_size, epoch):
    string_len = np.fromiter((len(string) for string in data), int)
    sorted_index = np.argsort(string_len)
    sorted_data = data[sorted_index]
    sorted_len = string_len[sorted_index]
    batch_size = {'mid': batch_size,
                  'true': batch_size * 8 / 9}
    
    def split_sum(c_sum, start_i):
        small_batch_i = []
        c_qt, c_rem = divmod(c_sum, round(batch_size['mid']))
        if c_qt == 0:
            c_qt = 1 
        else:
            if c_rem/c_qt > batch_size['mid']/2:
                c_qt = c_qt + 1
        b_size, b_rem = divmod(c_sum , c_qt)
        for x in range(c_qt):
            stop_i = start_i + b_size
            if x >= b_rem:
                stop_i -= 1
            small_batch_i.append((start_i, stop_i))
            start_i = stop_i + 1  
        return small_batch_i
        
    _, indexes, counts = np.unique(sorted_len,
                                   return_counts=True,
                                   return_index=True)
    batch_index =[]
    c_sum = 0
    for i, c in zip(indexes, counts):
        if c_sum == 0:
            start_i = i
        c_sum += c
        if c_sum >= batch_size['true']:
            batch_index.extend(split_sum(c_sum, start_i))
            c_sum = 0
    if c_sum > 0:
        last_start, last_stop = batch_index[-1]
        replace_sum = c_sum + last_stop - last_start + 1
        batch_index.extend(split_sum(replace_sum, last_start))
    
    for group_num in random.sample(range(len(batch_index)), epoch):
        i_range = batch_index[group_num]
        batch = [sorted_data[i] for i in range(i_range[0], i_range[1] + 1)]
        yield batch

        #pad_bat = pad_sequences(batch, **kwargs)       
