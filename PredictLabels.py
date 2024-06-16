import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pickle


# Custom Tokenizer and padding functions
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
        return [[self.word_index[word] for word in word_tokenize(text.lower()) if word in self.word_index] for text in
                texts]


def pad_sequences(sequences, maxlen, padding='pre'):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        if padding == 'pre':
            padded_sequences[i, -len(seq):] = np.array(seq)[:maxlen]
        elif padding == 'post':
            padded_sequences[i, :len(seq)] = np.array(seq)[:maxlen]
    return padded_sequences


# Define the model structure
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


# Function to load the trained model and encoders
def load_model_and_encoders(model_path, tokenizer_path, encoders_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load the label encoders
    with open(encoders_path, 'rb') as handle:
        label_encoders = pickle.load(handle)

    # Define the model structure and load the trained weights
    total_words = tokenizer.vocab_size
    embed_size = 100
    hidden_size = 150
    num_bands = len(label_encoders['Band'].classes_)
    num_genres = len(label_encoders['Genre'].classes_)
    num_subgenres = len(label_encoders['Subgenre'].classes_)
    max_sequence_len = 100

    model = LyricsClassificationModel(total_words, embed_size, hidden_size, max_sequence_len, num_bands, num_genres,
                                      num_subgenres).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, tokenizer, label_encoders, device


# Function to predict genre, subgenre, and band
def predict_genre_subgenre_band(text, model, max_sequence_len, tokenizer, label_encoders, device):
    model.eval()
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
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


# Example usage
model_path = 'best_lyrics_classification_model.pth'
tokenizer_path = 'tokenizer_claassif.pkl'
encoders_path = 'label_encoders.pkl'

model, tokenizer, label_encoders, device = load_model_and_encoders(model_path, tokenizer_path, encoders_path)

text_to_predict = "When the quest is over and the battle's won There's a land far to the south where we go to have some fun The wenches they are plenty, the alcohol is free The party lasts all through the night and the alcohol is free"
predicted_band, predicted_genre, predicted_subgenre = predict_genre_subgenre_band(text_to_predict, model, 100,
                                                                                  tokenizer, label_encoders, device)
print(f"Predicted Band: {predicted_band}, Predicted Genre: {predicted_genre}, Predicted Subgenre: {predicted_subgenre}")
