import os
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='PyTorch SBIR')
parser.add_argument('--meta-dir', type=str,
                    default='/your/path/DomainNet/preprocessing',
                    help='meta info dir with all txt files')
args = parser.parse_args()


DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']


def read_text(fpath):
    with open(fpath) as f:
        content = f.readlines()
    return content


def parse_paths(fpath):
    content = read_text(fpath)
    paths = []
    categories = []
    for line in content:
        paths.append(line.split()[0])
        categories.append(paths[-1].split('/')[1])
    return paths, categories


# collect all file paths
df = []
for d in DOMAINS:
    print('%s: Parsing training paths...' % (d))
    paths, categories = parse_paths(os.path.join(args.meta_dir, d + '_train.txt'))
    n_samples = len(paths)
    df_train = pd.DataFrame(
        data={'cat': categories, 'domain': [d] * n_samples,
              'split': ['train'] * n_samples},
        index=paths)

    print('Parsing testing paths...')
    paths, categories = parse_paths(os.path.join(args.meta_dir, d + '_test.txt'))
    n_samples = len(paths)
    df_test = pd.DataFrame(
        data={'cat': categories, 'domain': [d] * n_samples,
              'split': ['test'] * n_samples},
        index=paths)

    df.append(df_train)
    df.append(df_test)

# saving into sketch and images dataframes
df = pd.concat(df)
cond = df['domain'] == 'quickdraw'
df_sk = df.loc[cond]
df_im = df.loc[~cond]

df_sk.to_hdf('sk.hdf5', 'DomainNet')
df_im.to_hdf('im.hdf5', 'DomainNet')

# select categories that do not overlap with ImageNet
overlap = [
    # furniture 17
    'bathtub', 'bench', 'chair',
    'couch', 'fence', 'hot_tub',
    'mailbox', 'pillow', 'sleeping_bag',
    'streetlight', 'stove', 'swing_set',
    'table', 'teapot', 'toilet',
    'vase', 'umbrella',
    # mammals 17
    'bear', 'cat', 'camel',
    'dog', 'elephant', 'hedgehog',
    'kangaroo', 'lion', 'monkey',
    'panda', 'pig', 'rabbit',
    'sheep', 'squirrel', 'tiger',
    'whale', 'zebra',
    # tool 14
    'basket', 'bottlecap', 'broom',
    'bucket', 'drill', 'dumbbell',
    'hammer', 'nail', 'rifle',
    'screwdriver', 'shovel', 'stethoscope',
    'syringe', 'wheel',
    # clothes 10
    'bowtie', 'helmet', 'lipstick',
    'necklace', 'purse', 'shoe',
    'sock', 'sweater', 't-shirt',
    'underwear',
    # electricity 15
    'camera', 'cell_phone', 'computer', 'dishwasher',
    'fan', 'keyboard', 'laptop',
    'microphone', 'microwave', 'radio',
    'remote_control', 'telephone', 'television',
    'toaster', 'washing_machine',
    # buildings 7
    'barn', 'bridge', 'castle', 'church',
    'jail', 'lighthouse', 'tent',
    # office 10
    'backpack', 'binoculars', 'candle',
    'clock', 'cup', 'coffee_cup',
    'envelope', 'eraser', 'mug',
    'paintbrush',
    # road transportation 12
    'ambulance', 'bicycle', 'bus',
    'car', 'firetruck', 'motorbike',
    'pickup_truck', 'school_bus', 'tractor',
    'train', 'truck', 'van',
    # food 5
    'hot_dog', 'ice_cream', 'lollipop',
    'pizza', 'popsicle',
    # cold blooded 9
    'crab', 'crocodile', 'fish',
    'lobster', 'scorpion', 'shark',
    'snail', 'snake', 'spider',
    # instrument 9
    'cello', 'drums', 'guitar',
    'harp', 'piano', 'saxophone',
    'trombone', 'trumpet', 'violin',
    # fruit 3
    'banana', 'pineapple', 'strawberry',
    # sport 5
    'baseball', 'basketball', 'hockey_puck',
    'snorkel', 'soccer_ball',
    # bird 5
    'bird', 'flamingo', 'owl',
    'penguin', 'swan',
    # vegetable 3
    'broccoli', 'carrot', 'mushroom',
    # kitchen 5
    'frying_pan', 'hourglass', 'lighter',
    'wine_bottle', 'matches',
    # water transportation 4
    'aircraft_carrier', 'canoe', 'speedboat', 'submarine',
    # sky transportation 1
    'parachute',
    # insect 3
    'ant', 'bee', 'butterfly',
    # others 3
    'cannon', 'teddy-bear', 'traffic_light']

categories = df_im['cat'].unique()
not_overlap = [c for c in categories if c not in overlap]

# make sure there is at least 40 samples in every cetegory
pool = []
for cc in not_overlap:
    tmp = []
    for mm in df['domain'].unique():
        cond = (df['domain'] == mm) & (df['cat'] == cc)
        tmp.append(sum(cond))
    if np.min(tmp) < 40:
        pass
    else:
        pool.append(cc)

# select 45 categories for zero-shot
np.random.seed(1234)

selection = np.random.choice(pool, size=45, replace=False)
selection.sort()
print(selection)

np.save('zero_cnames.npy', selection)
