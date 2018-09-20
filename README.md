## CLUSE: Cross-Lingual Unsupervised Sense Embeddings
![Model](https://i.imgur.com/IEhTXFU.png)

CLUSE is an unsupervised learning framework for *crosslingual sense embeddings*, whose goal is to provide the community with:
* state-of-the-art multilingual sense embeddings where the embeddings are aligned in a common space
* large-scale and high-quality English-Chinese contextual similarity evaluation dataset

## Dependencies
* Python 2.7/3.6 with [NumPy](http://www.numpy.org/)/[SciPy](https://www.scipy.org/)
* [opencc-python-reimplemented](https://github.com/yichen0831/opencc-python)
* [zhon](https://zhon.readthedocs.io/en/latest/)
* [jieba](https://github.com/fxsjy/jieba)
* [nltk](https://www.nltk.org/)
* [Tensorflow 1.10](https://www.tensorflow.org/versions/r1.10/) with CUDA 9.0 and CuDNN v7.0.5

## Get training & evaluation datasets
Get training dataset for Engilsh-German parallel corpus: [Europarl](http://www.statmt.org/europarl/).

Get training dataset for English-Chinese parallel corpus: [UM-Corpus](http://nlp2ct.cis.umac.mo/um-corpus/).

Get mono-lingual sense embeddings evaluation dataset: [SCWS](https://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes).

Get cross-lingual sense embeddings evaluation dataset: [BCWS](https://github.com/MiuLab/BCWS).

Please cite the corresponding papers if you use the above datasets.

## Data preprocessing
All the data are in the data/ directory. You can safely download the preprocessed data from [here]().

Or you can preprocess the data by yourself.

First put the *dataset.txt* and *bcws.txt* from [BCWS](https://github.com/MiuLab/BCWS) into *data/en_ch/*.
Then put the *ratings.txt* from [SCWS](https://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes) into *data/en_ch/* and *data/en_de/*.

Since this work requires parallel corpus, you have to prepare two files for each language pair. These two files should have the same number of lines, such that the sentences with same line number form a paralle setence pair.

For example, to prepare the training and evaluation data for the Engilsh-German language pair,
```
cd data/en_ch/
bash run.sh english_parallel german_parallel english_vocab_size german_vocab_size
```
To reproduce the results in the [paper](https://arxiv.org/abs/1809.05694),
```
bash run.sh europarl-v7.de-en.en europarl-v7.de-en.de 6000 6000
```
will generate all the training and evaluation files.

Similarly,
```
cd data/en_ch/
bash run.sh en.txt ch.txt 6000 6000
```
will generate all the training and evaluation files for the Engilsh-Chinese language pair. Note that there are several domains in [UM-Corpus](http://nlp2ct.cis.umac.mo/um-corpus/), and we simply concatenate all the files.

## Training
To train the Engilsh-German sense embeddings:
```
cd en_de/
bash train.sh checkpoint_dir major_weight reg_weight
```
For example,
```
bash train.sh log 0.5 1.0
```
will train the model and save the checkpoint files to *log* directory with the specified major weight and regularization weight. For details, please refer to the [paper](https://arxiv.org/abs/1809.05694).

Similarly,
```
cd en_ch/
bash train.sh checkpoint_dir major_weight reg_weight
```
will train the model for the English-Chinese sense embeddings.

## Evaluation
You will see the spearman correlation score of SCWS/BCWS during the training process.

To evaluate the trained models:
```
cd en_de/ or cd en_ch/
bash dump.sh path_to_ckpt
```
will evaluate the SCWS/BCWS again and dump the trained sense embeddings.

To decode the sense for a specific word with its context,
```
cd en_de/ or cd en_ch/
bash decode.sh path_to_ckpt
```
Note that we only allow for English input currently.

## References
Please cite [[1]]() if you found the resources in this repository useful and cite [[2]]() if you use the BCWS dataset.

### CLUSE: Cross-Lingual Unsupervised Sense Embeddings

[1] Ta-Chung Chi and Yun-Nung Chen, [*CLUSE: Cross-Lingual Unsupervised Sense Embeddings*](https://arxiv.org/abs/1809.05694)

```
@inproceedings{chi-chen:2018:EMNLP2018,
  author    = {Chi, Ta-Chung  and  Chen, Yun-Nung},
  title     = {Cluse: Cross-lingual underspervised sense embeddings},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing(EMNLP)},
  month     = {October},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
}
```

### BCWS: Bilingual Contextual Word Similarity

[2] Ta-Chung Chi, Ching-Yen Shih and Yun-Nung Chen, [*BCWS: Bilingual Contextual Word Similarity*]()

```
@article{bcws,
  title={BCWS: Bilingual Contextual Word Similarity},
  author={Ta-Chung Chi, Ching-Yen Shih and Yun-Nung Che},
  journal={arXiv preprint arXiv:},
  year={2018}
}
```

