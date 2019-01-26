import pandas as pd
import pymongo
import pickle

client = pymongo.MongoClient()

cursor = client.guitar.chordsinkey.find({})

doc_list = []

for doc in cursor:
    doc_list.append(pd.Series(doc['chords']))

def list_split(chord_bass):
    if(len(chord_bass) == 1):
        chord_bass = [chord_bass[0], '']
    return pd.Series(chord_bass)

#returns dataframe splitting roots,chords, basses
def splitter(chords):
    chord_list = chords.str.split("_", expand=True)
    chord_list_2 = chord_list[1].str.split("/")
    chord_list_2 = chord_list_2.apply(list_split)
    return pd.concat([chord_list[0], chord_list_2], axis=1)
    
splitted_chords = pd.Series(doc_list).apply(splitter)

vocab_root = set()
vocab_harm = set()
vocab_bass = set()

for df in splitted_chords:
    df.columns = [0,1,2]
    vocab_root.update(df[0])
    vocab_harm.update(df[1])
    vocab_bass.update(df[2])

vocab_root = {i:x for x, i in enumerate(vocab_root)}
vocab_harm = {i:x+len(vocab_root) for x, i in enumerate(vocab_harm)}
vocab_bass = \
    {i:x+len(vocab_root)+len(vocab_harm) for x, i in enumerate(vocab_bass)}
    
vocab_dict = {'root':vocab_root , 'harm':vocab_harm, 'bass':vocab_bass}

vocab_index = []
for df in splitted_chords:
    temp_df = pd.DataFrame()
    for idx, vocab in zip(df,(vocab_root, vocab_harm, vocab_bass)):
        temp_df[idx] = [vocab[i] for i in df[idx]]
    vocab_index.append(temp_df)
    
pickle.dump(vocab_index, open( "vocabIndex.p", "wb" ) )
pickle.dump(vocab_dict, open("vocabDict.p", "wb"))



