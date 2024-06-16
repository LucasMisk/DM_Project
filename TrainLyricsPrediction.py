import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from tqdm import tqdm  # Import tqdm for the progress bar

# Load dataset
df = pd.read_csv('dataset.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Concatenate band name, genre, and subgenre with the lyrics
df['Input'] = df['Band'] + ' <genre> ' + df['Genre'] + ' <subgenre> ' + df['Subgenre'] + ' <lyrics> ' + df['Lyrics']

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

# Tokenize the input data
max_vocab_size = 5000  # Limit the vocabulary size
tokenizer = CustomTokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(df['Input'])
total_words = tokenizer.vocab_size

input_sequences = []
for line in df['Input']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = 100  # Limit the maximum sequence length
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
X, y = input_sequences[:,:-1], input_sequences[:,-1]

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

class LyricsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

dataset = LyricsDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

embed_size = 100
hidden_size = 150

model = LyricsModel(total_words, embed_size, hidden_size, max_sequence_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
best_loss = float('inf')
best_model_path = 'best_lyrics_model.pth'

import pickle
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(dataloader)
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f} (Best Model Saved)")

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

# Generating lyrics
def generate_lyrics(seed_text, next_words, model, max_sequence_len, tokenizer):
    model.eval()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        token_list = torch.tensor(token_list).long().to(device)
        with torch.no_grad():
            predicted = model(token_list)
            predicted_index = torch.argmax(predicted, dim=1).item()
        output_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + output_word
    return seed_text

# Loading the best model
best_model = LyricsModel(total_words, embed_size, hidden_size, max_sequence_len).to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

# Generate lyrics using the best model
seed_text = "Metallica"
generated_lyrics = generate_lyrics(seed_text, 100, best_model, max_sequence_len, tokenizer)
print(generated_lyrics)
