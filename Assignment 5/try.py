import sys
import os

from PIL import Image

path = "D:\Img\chars74k-lite"
all_imgs = []


def main():
    for x in os.walk(path):
        for y in x[2]:
            current = list(Image.open(f"{x[0]}\{y}").getdata())
            all_imgs.append((current, x[0][-1]))
        

if __name__ == "__main__":
    main()