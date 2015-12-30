"""File to download images to generate the datasets (baubles, christams trees
and snowy landscapes). Images are saved to 'downloaded/<dataset>/'."""
from __future__ import print_function
import time
import os
import socket
socket.setdefaulttimeout(10)
import urllib

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
WRITE_MAIN_DIR = os.path.join(FILE_DIR, "downloaded/")
URL_LISTS = [
    ("baubles", os.path.join(FILE_DIR, "baubles_urls.txt")),
    ("christmas-trees", os.path.join(FILE_DIR, "christmas-trees_urls.txt")),
    ("snowy-landscapes", os.path.join(FILE_DIR, "snowy-landscapes_urls.txt"))
]
INTERVAL = 1.0 # download an image every x seconds

def main():
    """Downloads images to generate the datasets."""
    for list_name, list_filename in URL_LISTS:
        print("----------------")
        print("-- Downloading images for: %s" % (list_name,))
        print("----------------")

        urls = open(list_filename, "r").readlines()
        urls = [url.strip() for url in urls]
        for url_idx, url in enumerate(urls):
            print("[%s] Downloading '%s' (%d of %d)" % (list_name, url, url_idx+1, len(urls)))
            try:
                download_image(url, os.path.join(WRITE_MAIN_DIR, list_name))
            except Exception as exc:
                print(exc)
            time.sleep(INTERVAL)

    print("Finished.")

def download_image(source_url, dest_dir):
    """Downloads an image from the source url and saves it to the directory.
    Images that were already downloaded are skipped automatically.

    Args:
        source_url The URL of the image.
        dest_dir The directory to save the image in.
    Returns:
        True if the image was downloaded
        False otherwise (including images that were skipped)
    """
    # image url to filepath
    index = source_url.rfind(".com/")
    image_name = source_url[index+len(".com/"):].replace("/", "-")
    filepath = os.path.join(dest_dir, image_name)

    if os.path.isfile(filepath):
        # skip image that was already downloaded
        print("[Info] Skipped image '%s', was already downloaded" % (source_url))
        return False
    else:
        # create directory if it doesnt exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        # download the image
        urllib.urlretrieve(source_url, filepath)
        return True

if __name__ == "__main__":
    main()
