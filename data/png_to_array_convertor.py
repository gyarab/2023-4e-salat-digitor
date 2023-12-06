from PIL import Image


def main():
    image = Image.open('train/0/0\\0.png').convert("LA")
    pixels = []
    for pixel in image.getdata():
        pixels.append(pixel[1])
    print(len(pixels))
    print(28*28)


if __name__ == "__main__":
    main()
