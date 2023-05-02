import requests
from bs4 import BeautifulSoup
import lyricsgenius as lg
from  tqdm import tqdm
import json
import os
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="lets get rolling!")

	parser.add_argument(
	"--start", type=int, default=0
	)
	parser.add_argument(
	"--end", type=int, default=2000
	)
	parser.add_argument(
		"--path", type=str, default='/graphics/scratch2/staff/Hassan/genius_crawl/genius_data/'
	)

	parser.add_argument(
		"--genius_token", type=str, default='?'
	)
	args = parser.parse_args()
	return args

def artists_all(show_url=False):
	'''Crawls the site for all the artists' urls'''
	artists_out = []
	letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0']
	for lett in letters:
		url = 'https://genius.com/artists-index/'+lett
		if show_url:
			print(url)
		r = requests.get(url)
		soup = BeautifulSoup(r.text, 'html.parser')
		artists = soup.find_all(name='a',attrs={'class':"artists_index_list-artist_name"})+soup.find_all(name='ul',attrs={'class':"artists_index_list"})[1].find_all(name='a')
		artists_out += [link['href'] for link in artists]
	return artists_out


def main():

	args = parse_args()
  
  # the process freezes for this artist
	to_remove=['France gall']
	myToken = args.genius_token
	path = args.path
	# get all the artists name
	artists = artists_all()
	artists = [u.split('/')[-1].replace('-', ' ') for u in artists]
	genius = lg.Genius(myToken, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"],
					   remove_section_headers=True, timeout=5)

	for r in to_remove:
		artists.remove(r)
	# in case we need to resume crawling again
	prepared = []
	for name in os.listdir(path):
		prepared.append(name[:-5])


	# get the remaining artists
	if len(prepared) > 1:
		remaining_artists = []
		for artist in artists:
			if artist not in prepared:
				remaining_artists.append(artist)

		artists = remaining_artists

	fail_cases = []
	for artist in tqdm(artists[args.start:args.end]):
		try:
			results = genius.search_artist(artist, max_songs=None)
			results.save_lyrics(filename='{}{}'.format(path, artist), overwrite=True, sanitize=False, verbose=False)

		except:
			fail_cases.append(artist)
			print(f"some exception with {artist}")

	with open("fail_cases_{}_to_{}".format(args.start,args.end), "w") as fp:
		json.dump(fail_cases, fp)

	print('missed {} out of {}'.format(len(fail_cases), len(artists)))


if __name__ == "__main__":
    main()
