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
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


def read_data(dataset_file, include_langs=None, exclude_langs=None):
    assert include_langs or exclude_langs
    assert not (include_langs and exclude_langs)
    df = pd.read_csv(dataset_file, index_col=None)
    if include_langs:
        df = df[df.language.isin(include_langs)]
    else:
        df = df[~df.language.isin(exclude_langs)]
    return df.text.values, df.emotion.values, df.language.values


class RandomSubsetSampler(Sampler):
    def __init__(self, indices, num_samples_per_lang):
        self.indices = indices
        self.num_samples = len(indices) * num_samples_per_lang
        self.subset = np.random.permutation(np.concatenate(
           [np.random.choice(x, num_samples_per_lang) for x in self.indices]
        ))

    def __iter__(self):
        subset = np.random.permutation(self.subset)
        return iter(subset.tolist())

    def __len__(self):
        return self.num_samples


Sample = namedtuple("Sample", [
    "input_ids",
    "input_mask",
    "segment_ids",
    "label_id",
    "lang"
])


def to_sample(text, label, lang, tokenizer, max_seq_length):
    # Tokenize and convert to ids
    input_tokens = tokenizer.tokenize(text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(input_tokens) > max_seq_length - 2:
        input_tokens = input_tokens[:(max_seq_length - 2)]

    # Add CLS and SEP tokens
    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    segment_ids = [0 for i in range(len(input_ids))]

    # Initialize padding mask
    input_mask = [1] * len(input_ids)

    # Pad to max_seq_length
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    return Sample(
        torch.tensor(input_ids),
        torch.tensor(input_mask),
        torch.tensor(segment_ids),
        torch.tensor(label, dtype=torch.float),
        lang
    )
