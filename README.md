# universal-joy

## A Dataset and Results for Classifying Emotions Across Languages

### How to cite

**DRAFT - to be revised**
```
@inproceedings{lamprinidis2021universal,
  title={Universal Joy A Dataset and Results for Classifying Emotions Across Languages},
  author={Lamprinidis, Sotiris and Bianchi, Federico and Hardt, Daniel and Hovy, Dirk},
  year={2021},
  volume={11th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA 2021)}
  organization={Association for Computational Linguistics}
}
```
### How to run

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
./extract_language.py dataset/low-resource.csv nl.csv nl

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
