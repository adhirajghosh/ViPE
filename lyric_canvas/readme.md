# LyricCanvas Dataset
- The lyricCanvas dataset contains approximately 10M lines of lyrics with corresponding visual elaborations (visualizable prompts).
- It could be used to train large language models to translate highly abstract concepts and metaphorical
phrases to visualizable prompts for image generation
- Due to copy right policies, we are no allowed to publish the lyrics, however, we release the visual elaborations and the scraper through which
you can collect the lyrics
## Compiling LyricCanvas
Building LyricCanvas involves two main steps:
- Step 1: Scraping lyrics from the Genius platform
- Step 2: Generating synthetic visual elaboration using a large language models (e.g., GPT3.5 Turbo)

### Step 1: Scraping Genius Lyrics
The first part of creating the LyricCanvas dataset is to scrape the lyrics form the
genius platform and convert them into a manageable dataset

Run the `scraper.py` script with the following arguments to scrape lyrics:

```bash
python scraper.py --start <start_index> --end <end_index> --path <output_path> --genius_token <your_genius_token>
```
- `--start`: Optional. Start index for scraping artists, default=0.
- `--end`: Optional. End index for scraping artists, default=-1.
- `--path`: Required. A directory, where the scraped lyrics will be saved.
- `--genius_token`: Required. Your Genius API token.

After scraping the lyrics, preprocess them and save as a single dataset using the prepare_lyrics.py script:

```bash
python prepare_lyrics.py --path <scraped_data_path> --path_out <output_path> --genius_token <your_genius_token>
```
- `--path`: Required. Path where the scraped lyrics are saved.
- `--path_out`: Required. Path to save the prepared lyrics dataset.
- `--genius_token`: Required. Your Genius API token.\
See other options using `--help`\
The resulting dataset is saved as a dictionary file with the following structure:
```python
dict = {
    "artist_name": [
        { "title": "song1", "lyrics": "l1", ... },
        { "title": "song2", "lyrics": "l2", ... }
    ],
    ...
}
```
#### Viewing Dataset Statistics
The info.py script provides statistics about the prepared dataset:
```bash
python info.py --path_out <prepared_data_path> --max_lines <max_lines_per_lyric>
```
- `--path_out`: Path to the prepared lyrics dataset.
- `--max_lines`: Maximum number of lines per lyric used for preparing the lyrics.\
---
**Our version of the genius lyrics contain the following statistics :**
- number of artists:  5549
- total songs:  249948
- total lines:  9909617
- mean , std  lines per track:  39.646714516619454   10.77...
- min , max  lines per track:  14   50
- number of bad name or title 0
---

### Step 2: Generating Visual Elaborations
Once you have the lyrics prepared, you could either:
- Option 1: Use our generated visual elaborations available [here](www.googlecmo).
- Option 2: Generate new visual elaborations for your own set of lyrics 

### Option 1: Use our generated visual elaborations 

```python
dict = {
    "artist_name": [
        { "title": "song1", "lyrics": "l1", ... },
        { "title": "song2", "lyrics": "l2", ... }
    ],
    ...
}
```