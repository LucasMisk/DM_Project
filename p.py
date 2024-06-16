import requests
from bs4 import BeautifulSoup

# Your Genius API access token
access_token = 'G0AyWjEdB67AFEr9byFlU3bRZMATSHseZFz1mezUllAFVs89AIhQCv8g95HNZhUS'

# Function to search for a song on Genius
def search_song(song_title, artist_name):
    base_url = "https://api.genius.com"
    headers = {'Authorization': 'Bearer ' + access_token}
    search_url = base_url + "/search"
    data = {'q': song_title + ' ' + artist_name}
    response = requests.get(search_url, headers=headers, params=data)
    return response.json()

# Function to get song URL from the search results
def get_song_url(search_results):
    for hit in search_results['response']['hits']:
        if hit['result']['primary_artist']['name'].lower() == artist_name.lower():
            return hit['result']['url']
    return None

# Function to scrape song lyrics from Genius
def scrape_lyrics(song_url):
    page = requests.get(song_url)
    html = BeautifulSoup(page.text, 'html.parser')
    divs = html.find_all('div', class_='Lyrics__Container-sc-1ynbvzw-1 kUgSbL')
    print(len(divs))
    lyrics = ""
    for div in divs:
        lyrics += div.get_text(strip=False)  # Extract and clean the text

    return lyrics

# Define the song and artist
song_title = "Shape of You"
artist_name = "Ed Sheeran"

# Search for the song
search_results = search_song(song_title, artist_name)

# Get the song URL
song_url = get_song_url(search_results)

if song_url:
    # Scrape the lyrics
    lyrics = scrape_lyrics(song_url)
    print(lyrics)
else:
    print("Song not found")