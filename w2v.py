import ipdb
import numpy as np
import gensim.downloader as api
import pandas as pd


# manually curated names to match word2vec entries
NAMES = {'axe': ['ax'],
         'coffee_cup': ['coffee_mug'],
         'hot_dog': ['hotdog'],
         'house_plant': ['houseplant'],
         'paper_clip': ['paperclip'],
         'school_bus': ['schoolbus'],
         'soccer_ball': ['football'],
         'swing_set': ['swingset'],
         'The_Eiffel_Tower': ['Eiffel_Tower'],
         'moustache': ['mustache'],
         'see_saw': ['seesaw'],
         'string_bean': ['green_beans'],
         'The_Mona_Lisa': ['Mona_Lisa'],
         'wine_glass': ['wineglass'],
         'The_Great_Wall_of_China': ['Great_Wall', 'China']}


def get_vector_names(classnames):
    print('Loading word2vec...')
    model = api.load("word2vec-google-news-300")

    wv = {}
    for cls in classnames:
        # print(cls)
        tmp = cls.replace('-', '_')
        try:
            vec = model.get_vector(tmp)
        except:
            if tmp in NAMES:
                vec = np.mean([model.get_vector(w) for w in NAMES[tmp]], axis=0)
            else:
                vec = np.mean([model.get_vector(w) for w in tmp.split('_')], axis=0)
        wv[cls] = vec
    return wv


df = pd.read_hdf('im.hdf5')
classnames = df['cat'].unique()

w2v = get_vector_names(classnames)
np.save('w2v.npy', w2v)
