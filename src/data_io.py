# +
import os
from skimage import io


def imread(fullpath):
    return io.imread(fullpath)


def readfile(path):
    print(f'Reading file at {path}')
    with open(path, 'r') as f:
        lines = f.read()
        print(lines)
        
def savefig(fig, savepath, filename, ):
    if not(path is None):
        filename = 'PMT nonlinearity from '+str(label)
        subdir = os.path.join(current_data_dir, subdir_name)
        if not(os.path.isdir(subdir)):
            os.makedirs(subdir)
        pathname = os.path.join(subdir, filename)
        fig.savefig(pathname)


