# Universal Joy
### A Dataset and Results for Classifying Emotions Across Languages

by Me and [Federico Bianchi](https://federicobianchi.io),
[Daniel Hardt](https://www.cbs.dk/en/research/departments-and-centres/department-of-management-society-and-communication/staff/dhamsc),
[Dirk Hovy](http://www.dirkhovy.com).

The dataset is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

-----
## Abstract

> While emotions are universal aspects of human psychology, they are expressed differently across different languages and cultures.
> We introduce a new data set of over 530k anonymized public Facebook posts across 18
> languages, labeled with five different emotions. Using multilingual BERT embeddings,
> we show that emotions can be reliably inferred both within and across languages. Zero-shot
> learning produces promising results for low resource languages. Following established theories of basic emotions, we provide a detailed
> analysis of the possibilities and limits of cross-lingual emotion classification. We find that
> structural and typological similarity between languages facilitates cross-lingual learning, as
> well as linguistic diversity of training data.  Our results suggest that there are commonalities underlying the
> expression of emotion in different languages. We publicly release the anonymized data for future research.


## How to cite

**To be revised!!!**
```
@inproceedings{lamprinidis2021universal,
  title={Universal Joy A Dataset and Results for Classifying Emotions Across Languages},
  author={Lamprinidis, Sotiris and Bianchi, Federico and Hardt, Daniel and Hovy, Dirk},
  year={2021},
  volume={11th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA 2021)}
  organization={Association for Computational Linguistics}
}
```
## How to run

```bash
# Prepare the data
tar xf emotion-datasets.tar.gz
ln -s datasets/small.csv emotion-train.csv
ln -s datasets/val.csv emotion-val.csv
ln -s datasets/test.csv emotion-test.csv

pip install -r requirements.txt


# Train a model on the small english subset and validate on english
./script.py --train_languages en --val_languages en --save en-small.pt

# Extract the dutch low-resource set
./extract_language.py datasets/low-resource.csv nl.csv nl

# Test the model we've just trained on that
./test.py nl.csv en-small.pt en-to-nl

  macro f1: 0.39137742214549537
  micro f1:0.5485151659520408
  anger: 0.3605870020964361
  anticipation: 0.5790686952512678
  fear: 0.025974025974025972
  joy: 0.6198288159771754
  sadness: 0.37142857142857144
```

## Dataset splits
### Small (2,947 instances per language)

| language     |   en |   es |   pt |   tl |   zh |
|--------------|------|------|------|------|------|
| **emotion**  |      |      |      |      |      |
| anger        |  175 |  169 |  167 |  389 |  276 |
| anticipation | 1200 |  685 |  762 |  548 |  657 |
| fear         |   34 |   14 |   21 |   95 |   63 |
| joy          | 1153 | 1616 | 1257 | 1300 | 1600 |
| sadness      |  385 |  463 |  740 |  615 |  351 |

### Large (29,364 instances per language)

| language     |    en |    es |    pt |
|--------------|-------|-------|-------|
| **emotion**  |       |       |       |
| anger        |  1740 |  1683 |  1664 |
| anticipation | 11961 |  6829 |  7595 |
| fear         |   337 |   140 |   204 |
| joy          | 11488 | 16099 | 12527 |
| sadness      |  3838 |  4613 |  7374 |


### Huge (282,313 instances, only English)

| emotion      |   instances |
|--------------|-------------|
| anger        |       16726 |
| anticipation |      115000 |
| fear         |        3236 |
| joy          |      110446 |
| sadness      |       36905 |

### Low Resource (different size for every language)

| language     |   bn |   de |   fr |   hi |   id |   it |   km |   ms |   my |   nl |   ro |   th |   vi |
|--------------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| **emotion**  |      |      |      |      |      |      |      |      |      |      |      |      |      |
| anger        |  120 |  425 |  382 |  274 |  382 |  472 |  115 |  326 |  177 |  150 |   97 |  244 |  176 |
| anticipation |  211 | 1475 | 1788 |  231 | 1841 | 1910 |  158 | 1344 |  130 |  788 |  560 |  938 | 1137 |
| fear         |    7 |    8 |   22 |    8 |   32 |   20 |   23 |   34 |    9 |   10 |    8 |   21 |   39 |
| joy          |  249 | 3388 | 3222 |  830 | 3077 | 3656 |  469 | 2566 |  412 |  981 |  923 | 2202 | 1982 |
| sadness      |  282 |  606 | 1143 |  480 |  869 |  651 |  212 |  638 |  225 |  272 |  352 |  398 |  622 |

