from PIL import Image
import sys


def load_image(filename):
    image = Image.open(f"{filename}").convert("LA")
    pixels = ""
    for pixel in image.getdata():
        pixels += str(pixel[1]) + " "
    print(f"{pixels}")


if __name__ == "__main__":
    load_image(sys.argv[1])
