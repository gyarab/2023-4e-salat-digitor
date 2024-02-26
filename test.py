from PIL import Image
import subprocess
import sys


def test_all(filename):
    process = subprocess.Popen(["./digitor", f"{filename}"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    count = 0
    wrongs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(f"# Testing on training dataset")
    for i in range(0, 8000):
        for j in range(0, 10):
            image = Image.open(f'train/{j}/{j}\\{i}.png').convert("LA")
            pixels = ""
            for pixel in image.getdata():
                pixels += str(pixel[1]) + " "
            process.stdin.write(f"{pixels}\n")
            process.stdin.flush()
            output = process.stdout.readline().strip()
            if output != f"{j}":
                wrongs[j] += 1
                count += 1
    print(f"{count}/80000 wrong ({round(100 - count / 800, 1)}% correct)")
    print(wrongs)
    print()

    count = 0
    wrongs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(f"# Testing on testing dataset")
    for i in range(0, 2773):
        for j in range(0, 10):
            image = Image.open(f'test/{j}/{j}\\{i}.png').convert("LA")
            pixels = ""
            for pixel in image.getdata():
                pixels += str(pixel[1]) + " "
            process.stdin.write(f"{pixels}\n")
            process.stdin.flush()
            output = process.stdout.readline().strip()
            if output != f"{j}":
                wrongs[j] += 1
                count += 1
    print(f"{count}/27730 wrong ({round(100 - count / 277.3, 1)}% correct)")
    print(wrongs)


def main(filename):
    test_all(filename)


if __name__ == '__main__':
    main(sys.argv[1])
