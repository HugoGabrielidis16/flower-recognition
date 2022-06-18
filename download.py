from six.moves import urllib
import tarfile


if __name__ == "__main__":
    DOWNLOAD_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
    print("Downloading flower images from %s..." % DOWNLOAD_URL)
    urllib.request.urlretrieve(DOWNLOAD_URL, "flower_photos.tgz")

    tar = tarfile.open("flower_photos.tgz", "r:gz")
    tar.extractall()
    tar.close()
