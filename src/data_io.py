# +
import os

def readfile(path):
    print(f'Reading file at {path}')
    with open(path, 'r') as f:
        lines = f.read()
        print(lines)


