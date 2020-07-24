# domainnet-helpers

Helper scripts for open cross-domain retrieval experiments with the DomainNet dataset [[website](http://ai.bu.edu/M3SDA/)].  

## Image annotations

Run [parse.py](parse.py) to create pandas dataframes for all domains of the dataset. There will be 2 dataframes: one for the quickdraw domain, one for the five other domains. Each entry in the dataframe consist of the filename path, the label and the split.

[parse.py](parse.py) also contains the list of the 188 categories of DomainNet that overlap with ImageNet.

In our paper we consider the `quickdraw` domain as `sketches`, and the `sketch` domain as `pencil sketches`.

## Word vectors

Run [w2v.py](w2v.py) to get the word vector for all class names. Word vectors are stored in a dictionary in a `.npz` file.

It requires the gensim library: `conda install -c anaconda gensim`

## Image pre-processing

Run [resize.py](resize.py). Images from all domains will be resized to 224x224.

## Citation
If you find these scripts useful, please consider citing our paper:

```
@article{
    Thong2020OpenSearch,
    title={Open Cross-Domain Visual Search},
    author={Thong, William and Mettes, Pascal and Snoek, Cees G.M.},
    journal={CVIU},
    year={2020},
    url={https://arxiv.org/abs/1911.08621}
}
```
