import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('dataset.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Encode labels
label_encoders = {
    'Band': LabelEncoder(),
    'Genre': LabelEncoder(),
    'Subgenre': LabelEncoder()
}
df['Band'] = label_encoders['Band'].fit_transform(df['Band'])
df['Genre'] = label_encoders['Genre'].fit_transform(df['Genre'])
df['Subgenre'] = label_encoders['Subgenre'].fit_transform(df['Subgenre'])

# Concatenate band name, genre, and subgenre with the lyrics
df['Input'] = df['Band'].astype(str) + ' <genre> ' + df['Genre'].astype(str) + ' <subgenre> ' + df['Subgenre'].astype(str) + ' <lyrics> ' + df['Lyrics']

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

input_sequences = tokenizer.texts_to_sequences(df['Input'])
max_sequence_len = 100  # Limit the maximum sequence length
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
X = torch.tensor(input_sequences, dtype=torch.long)
y_band = torch.tensor(df['Band'].values, dtype=torch.long)
y_genre = torch.tensor(df['Genre'].values, dtype=torch.long)
y_subgenre = torch.tensor(df['Subgenre'].values, dtype=torch.long)

class LyricsDataset(Dataset):
    def __init__(self, X, y_band, y_genre, y_subgenre):
        self.X = X
        self.y_band = y_band
        self.y_genre = y_genre
        self.y_subgenre = y_subgenre

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_band[idx], self.y_genre[idx], self.y_subgenre[idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

dataset = LyricsDataset(X, y_band, y_genre, y_subgenre)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model
class LyricsClassificationModel(nn.Module):
    def __init__(self, total_words, embed_size, hidden_size, max_sequence_len, num_bands, num_genres, num_subgenres):
        super(LyricsClassificationModel, self).__init__()
        self.embedding = nn.Embedding(total_words, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc_band = nn.Linear(hidden_size, num_bands)
        self.fc_genre = nn.Linear(hidden_size, num_genres)
        self.fc_subgenre = nn.Linear(hidden_size, num_subgenres)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        band_out = self.fc_band(x)
        genre_out = self.fc_genre(x)
        subgenre_out = self.fc_subgenre(x)
        return band_out, genre_out, subgenre_out

embed_size = 100
hidden_size = 150
num_bands = len(label_encoders['Band'].classes_)
num_genres = len(label_encoders['Genre'].classes_)
num_subgenres = len(label_encoders['Subgenre'].classes_)

model = LyricsClassificationModel(total_words, embed_size, hidden_size, max_sequence_len, num_bands, num_genres, num_subgenres).to(device)
criterion_band = nn.CrossEntropyLoss()
criterion_genre = nn.CrossEntropyLoss()
criterion_subgenre = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
best_loss = float('inf')
best_model_path = 'best_lyrics_classification_model.pth'

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, labels_band, labels_genre, labels_subgenre in dataloader:
        inputs, labels_band, labels_genre, labels_subgenre = inputs.to(device), labels_band.to(device), labels_genre.to(device), labels_subgenre.to(device)
        outputs_band, outputs_genre, outputs_subgenre = model(inputs)
        loss_band = criterion_band(outputs_band, labels_band)
        loss_genre = criterion_genre(outputs_genre, labels_genre)
        loss_subgenre = criterion_subgenre(outputs_subgenre, labels_subgenre)

        loss = loss_band + loss_genre + loss_subgenre

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f} (Best Model Saved)")

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

# After training the model
with open('tokenizer_claassif.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoders.pkl', 'wb') as handle:
    pickle.dump(label_encoders, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the best model
best_model = LyricsClassificationModel(total_words, embed_size, hidden_size, max_sequence_len, num_bands, num_genres, num_subgenres).to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()

# Predict genre, subgenre, and band function
def predict_genre_subgenre_band(text, model, max_sequence_len, tokenizer, label_encoders):
    model.eval()
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    token_list = torch.tensor(token_list).long().to(device)
    with torch.no_grad():
        output_band, output_genre, output_subgenre = model(token_list)
        predicted_band_idx = torch.argmax(output_band, dim=1).item()
        predicted_genre_idx = torch.argmax(output_genre, dim=1).item()
        predicted_subgenre_idx = torch.argmax(output_subgenre, dim=1).item()
    predicted_band = label_encoders['Band'].inverse_transform([predicted_band_idx])[0]
    predicted_genre = label_encoders['Genre'].inverse_transform([predicted_genre_idx])[0]
    predicted_subgenre = label_encoders['Subgenre'].inverse_transform([predicted_subgenre_idx])[0]
    return predicted_band, predicted_genre, predicted_subgenre

# Example usage:
text_to_predict = "<lyrics> Some sample lyrics text"
predicted_band, predicted_genre, predicted_subgenre = predict_genre_subgenre_band(text_to_predict, best_model, max_sequence_len, tokenizer, label_encoders)
print(f"Predicted Band: {predicted_band}, Predicted Genre: {predicted_genre}, Predicted Subgenre: {predicted_subgenre}")
