from tensorflow.keras.utils import to_categorical
import numpy as np

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size):
    while True:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield [[photo, in_seq], out_seq]
