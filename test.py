#!/usr/bin/env python3
# Copyright © 2019 Sotiris Lamprinidis et.al.
#   part of the publication presented here <url>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse

import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import torch
from pytorch_transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
)

from script import BertForSequenceClassification
from utils import to_sample
"""
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)
"""

EMOTIONS = "anger anticipation fear joy sadness".split()


def read_data(fname):
    df = pd.read_csv(fname, index_col=None)
    y = np.zeros((len(df), 5))
    for i, emotions in enumerate(df.iloc[:, 1].values):
        for emotion in emotions.split():
            if emotion in EMOTIONS:
                y[i, EMOTIONS.index(emotion)] = 1.
    return df.iloc[:, 0].values, y


def main(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased",
                                              do_lower_case=False)
    bert_config = BertConfig.from_pretrained("bert-base-multilingual-cased")
    bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    model = BertForSequenceClassification(
        bert_model, bert_config, num_labels=5)

    model = model.cuda()
    model.load_state_dict(torch.load(args.model_fname))

    X_test, y_test = read_data(args.data_fname)

    test_samples = [to_sample(x, y, "nan", tokenizer, args.max_seq_length)
                    for x, y in zip(X_test, y_test)]

    # Test set
    y_pred = np.zeros_like(y_test)
    model.eval()
    for i, sample in enumerate(test_samples):
        inputs = {'input_ids':      sample[0].unsqueeze(0).cuda(),
                  'attention_mask': sample[1].unsqueeze(0).cuda(),
                  'token_type_ids': sample[2].unsqueeze(0).cuda(),
                  'labels':         sample[3].unsqueeze(0).cuda()}

        with torch.no_grad():
            logits = model(**inputs)
        y_pred[i] = logits.cpu().float()

    np.save(f"{args.out_fname}.npy", y_pred, allow_pickle=False)
    cols = ~(y_test == 0).all(0)
    y_test = y_test[:, cols]
    y_pred = y_pred[:, cols]
    fscores = f1_score(y_test, y_pred >= 0.0, average=None)
    fscore_micro = f1_score(y_test, y_pred >= 0.0, average="micro")
    print(f"macro f1: {float(np.mean(fscores))}")
    print(f"micro f1:{float(fscore_micro)}")
    for emotion, fscore in zip(EMOTIONS, fscores):
        print(f"{emotion}: {fscore}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_fname")
    parser.add_argument("model_fname")
    parser.add_argument("out_fname")
    parser.add_argument("-m", "--max_seq_length", type=int, default=256)

    main(parser.parse_args())
