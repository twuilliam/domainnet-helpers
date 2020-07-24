import os
import argparse
import multiprocessing as mp
import pandas as pd
from PIL import Image


# Training settings
parser = argparse.ArgumentParser(description='PyTorch SBIR')
parser.add_argument('--data-dir', type=str,
                    default='/your/path/DomainNet/',
                    help='DomainNet directory')

args = parser.parse_args()


def resize_img(path):
    full_path = os.path.join(args.data_dir, path)
    im = Image.open(full_path)
    im = im.convert('RGB')
    im = im.resize((224, 224), Image.ANTIALIAS)

    new_path = os.path.join(args.data_dir, 'resized', path)
    if os.path.splitext(new_path)[1] in ['png', 'PNG', 'JPEG', 'JPG']:
        new_path = os.path.splitext(new_path)[0] + '.jpg'
    directory = os.path.dirname(new_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    im.save(new_path, quality=95)


def resize_img_multi(keys):
    for k in keys:
        try:
            resize_img(k)
        except:
            print(k)


def worker(q, keys):
    q.put(resize_img_multi(keys))


NUM_WORKERS = 24

df1 = pd.read_hdf('im.hdf5')
df2 = pd.read_hdf('sk.hdf5')

df = pd.concat([df1, df2])

q = mp.Queue()
processes = []

paths = df.index.to_list()
n = len(paths)
for i in range(NUM_WORKERS):
    lower = int((i) * n / (NUM_WORKERS))
    upper = int((i + 1) * n / (NUM_WORKERS))
    processes.append(mp.Process(target=worker,
                                args=(q, paths[lower:upper])))
for p in processes:
    p.start()
for p in processes:
    p.join()
