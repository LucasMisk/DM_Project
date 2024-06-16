import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pickle

# Tokenizer and padding functions
class CustomTokenizer:
    def __init__(self, num_words=None):
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0
        self.num_words = num_words

    def fit_on_texts(self, texts):
        words = [word for text in texts for word in word_tokenize(text.lower())]
        freq_dist = FreqDist(words)
        if self.num_words:
            freq_dist = dict(freq_dist.most_common(self.num_words))
        self.word_index = {word: index + 1 for index, (word, _) in enumerate(freq_dist.items())}
        self.index_word = {index: word for word, index in self.word_index.items()}
        self.vocab_size = len(self.word_index) + 1

    def texts_to_sequences(self, texts):
        return [[self.word_index[word] for word in word_tokenize(text.lower()) if word in self.word_index] for text in texts]

def pad_sequences(sequences, maxlen, padding='pre'):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if padding == 'pre':
            padded_sequences[i, -len(seq):] = np.array(seq)[:maxlen]
        elif padding == 'post':
            padded_sequences[i, :len(seq)] = np.array(seq)[:maxlen]
    return padded_sequences

# Define the model
class LyricsModel(nn.Module):
    def __init__(self, total_words, embed_size, hidden_size, max_sequence_len):
        super(LyricsModel, self).__init__()
        self.embedding = nn.Embedding(total_words, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, total_words)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Generating lyrics with temperature control
def generate_lyrics(seed_text, next_words, model, max_sequence_len, tokenizer, temperature=1.0):
    model.eval()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        token_list = torch.tensor(token_list).long().to(device)
        with torch.no_grad():
            predicted = model(token_list)
            predicted = predicted / temperature
            probabilities = torch.nn.functional.softmax(predicted, dim=-1)
            predicted_index = torch.multinomial(probabilities, 1).item()
        output_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + output_word
    return seed_text

# Loading the best model and tokenizer
best_model_path = 'best_lyrics_model.pth'
tokenizer_path = 'tokenizer.pkl'

# Assuming 'tokenizer' is the instance of CustomTokenizer
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_words = tokenizer.vocab_size
embed_size = 100
hidden_size = 150
max_sequence_len = 100

best_model = LyricsModel(total_words, embed_size, hidden_size, max_sequence_len).to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

# Generate lyrics using the best model
seed_text = "Tyr <genre> Metal"
generated_lyrics = generate_lyrics(seed_text, 100, best_model, max_sequence_len, tokenizer, temperature=0.8)
print(generated_lyrics)
