from PIL import Image
import subprocess


def test_all():
    process = subprocess.Popen(["./digitor", "digitor.json"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    count = 0
    wrongs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
                #print(f"wrong: image: {j}/{j}\\{i}.png  output: {output}")
                wrongs[j] += 1
                count += 1
    print(f"{count} wrong")
    print(wrongs)


def main():
    test_all()


if __name__ == '__main__':
    main()
