from PIL import Image
import subprocess
import sys


def test_all(filename):
    process = subprocess.Popen(["./digitor", f"{filename}"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for dataset_path, size in [("data/train/", 80000), ("data/test/", 27730)]:
        print(f"# Testing \'{dataset_path}\' dataset")
        count = 0
        wrongs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(0, size // 10):
            for j in range(0, 10):
                image = Image.open(f"{dataset_path}{j}/{j}\\{i}.png").convert("LA")
                pixels = ""
                for pixel in image.getdata():
                    pixels += str(pixel[1]) + " "
                process.stdin.write(f"{pixels}\n")
                process.stdin.flush()
                output = process.stdout.readline().strip()
                if output != f"{j}":
                    wrongs[j] += 1
                    count += 1
        print(f"{count}/{size} wrong ({round(100 - (count / (size / 100)), 1)}% correct)")
        print(wrongs)
        print()


if __name__ == "__main__":
    test_all(sys.argv[1])
