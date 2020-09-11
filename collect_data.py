import pandas as pd 
import spotipy 
import spotipy.oauth2 as oauth2

def show_artist(artist):
	print('====', artist['name'], '====')
	print('Popularity: ', artist['popularity'])
	if len(artist['genres']) > 0:
		print('Genres: ', ','.join(artist['genres']))


def get_dataframe(name):
	names = sp.search(q='artist:' + name, type='artist')
	artist = names['artists']['items'][0]
	albums = []
	results = sp.artist_albums(artist['id'], album_type='album')
	albums.extend(results['items'])
	while results['next']:
		results = sp.next(results)
		albums.extend(results['items'])
	songs = set()
	features = []
	for album in albums:
		tracks = []	
		results = sp.album_tracks(album['id'])
		tracks.extend(results['items'])
		while results['next']:
			results = sp.next(results)
			tracks.extend(results['items'])
		for track in tracks:
			if(track['name'] not in songs):
				songs.add(track['name'])
				print("getting features for {}".format(track['name']))
				features.extend(sp.audio_features(track['id']))
		# break
	print(songs)
	df = pd.DataFrame(features)
	return df


credentials = oauth2.SpotifyClientCredentials(
        client_id="9474eabdc9f640a8bded6c379bd77e61",
        client_secret="9b60e5a336e54842a869fb60f6e78177")
token = credentials.get_access_token()
sp = spotipy.Spotify(auth = token)
sp.trace = False
#gathering data!!
rap_artists = ["Drake", "Eminem", "Kendrick Lamar"]
pop_artists = ["Ed Sheeran", "Selena Gomez", "Justin Bieber"]
rnb_artists = ["Daniel Caesar", "H.E.R.", "Kehlani"]

rap_df = pd.concat([get_dataframe(name) for name in rap_artists], axis = 0)
pop_df = pd.concat([get_dataframe(name) for name in pop_artists], axis = 0)
rnb_df = pd.concat([get_dataframe(name) for name in rnb_artists], axis = 0)

rap_df.to_csv('data_output/rap_data.csv')
pop_df.to_csv('data_output/pop_data.csv')
rnb_df.to_csv('data_output/rnb_data.csv')