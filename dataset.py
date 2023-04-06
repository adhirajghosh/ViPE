from urllib.request import urlretrieve


def load_dataset(url):


    urlretrieve(url, 'metadata.parquet')

def main():
    url = 'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata-large.parquet'