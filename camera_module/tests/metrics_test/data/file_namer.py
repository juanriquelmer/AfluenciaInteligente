# Generate a code that names the files in the folder like
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ...

import os
import sys

"""
def main():
    if len(sys.argv) != 2:
        print("Usage: python file_namer.py <folder>")
        return

    folder = sys.argv[1]
    os.chdir(folder)
    files = os.listdir()

    for i in range(len(files)):
        os.rename(files[i], str(i+1) + ".jpg")
"""
# Do with the files in this folder
def main():
    files = os.listdir()

    for i in range(len(files)):
        os.rename(files[i], str(i+1) + ".jpg")
        
if __name__ == "__main__":
    main()