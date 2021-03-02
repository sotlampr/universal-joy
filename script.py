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
from collections import defaultdict
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_transformers import (
    AdamW,
    BertTokenizer,
    BertModel,
    BertConfig,
    WarmupLinearSchedule
)
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder

from utils import read_data, to_sample, RandomSubsetSampler


class BertForSequenceClassification(nn.Module):
    def __init__(self, model, config, num_labels):
        super().__init__()
        self.bert = model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, position_ids=None, head_mask=None, langs=None):
        outputs = self.bert(input_ids, position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        return self.linear(self.dropout(outputs[1]))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main(args):
    set_seed(args)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased",
                                              do_lower_case=False)
    bert_config = BertConfig.from_pretrained("bert-base-multilingual-cased")
    bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    model = BertForSequenceClassification(
        bert_model, bert_config, num_labels=5)

    model = model.cuda()

    X_train, y_train, train_langs = read_data(
        "emotion-train.csv",
        include_langs=args.train_languages,
    )
    X_val, y_val, val_langs = read_data(
        "emotion-val.csv",
        include_langs=args.val_languages,
    )
    X_test, y_test, test_langs = read_data(
        "emotion-test.csv",
        include_langs=args.val_languages,
    )

    target_encoder = OneHotEncoder(sparse=False)
    y_train = target_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val = target_encoder.transform(y_val.reshape(-1, 1))
    y_test = target_encoder.transform(y_test.reshape(-1, 1))

    train_samples = [to_sample(x, y, z, tokenizer, args.max_seq_length)
                     for x, y, z in zip(X_train, y_train, train_langs)]
    val_samples = [to_sample(x, y, z, tokenizer, args.max_seq_length)
                   for x, y, z in zip(X_val, y_val, val_langs)]
    test_samples = [to_sample(x, y, z, tokenizer, args.max_seq_length)
                    for x, y, z in zip(X_test, y_test, test_langs)]

    sampler = None
    shuffle = True
    if args.limit is not None:
        shuffle = False
        indices = [np.argwhere(train_langs == lang).ravel()
                   for lang in np.unique(train_langs)]
        limit_per_lang = args.limit // (len(np.unique(train_langs)))
        sampler = RandomSubsetSampler(indices, limit_per_lang)

    train_loader = DataLoader(train_samples, args.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_samples, args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_samples, args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    n_val_steps = len(val_loader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    t_total = len(train_loader) * args.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Warmup: half epoch
    warmup_steps = len(train_loader) // 2
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                     t_total=t_total)

    # Half-precision
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level,
                                          verbosity=0)

    # Criterion  & class_weight
    pos_weight = (y_train != 0).sum(0)
    pos_weight = (y_train.shape[0] - pos_weight) / pos_weight
    pos_weight = torch.from_numpy(pos_weight).float().cuda()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    global_step = 1
    for epoch in range(1, args.num_train_epochs+1):
        model.train()
        tr_loss = 0.0
        tr_fscore = 0.0
        for step, batch in enumerate(train_loader, 1):
            inputs = {'input_ids':      batch[0].cuda(),
                      'attention_mask': batch[1].cuda(),
                      'token_type_ids': batch[2].cuda(),
                      'labels':         batch[3].cuda()}

            logits = model(**inputs)
            loss = criterion(logits, inputs["labels"])
            tr_fscore += f1_score(
                inputs["labels"].cpu(), logits.cpu() >= 0, average="macro"
            )
            model.zero_grad()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                               args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            global_step += 1
            print(
                f"\repoch {epoch} step {global_step} "
                f"loss {tr_loss/step:.3f} fscore {tr_fscore/step:.3f} ",
                end="", flush=True
            )

            if global_step % args.val_every == 0:
                # Evaluation
                model.eval()
                val_y_true = defaultdict(list)
                val_y_pred = defaultdict(list)
                val_loss = 0.0
                for batch in val_loader:
                    inputs = {'input_ids':      batch[0].cuda(),
                              'attention_mask': batch[1].cuda(),
                              'token_type_ids': batch[2].cuda(),
                              'labels':         batch[3].cuda()}
                    langs = batch[4]

                    with torch.no_grad():
                        logits = model(**inputs)
                        loss = criterion(logits, inputs["labels"])

                    for target, pred, lang in zip(inputs["labels"].cpu(),
                                                  logits.cpu() >= 0,
                                                  langs):
                        val_y_true[lang].append(target.tolist())
                        val_y_pred[lang].append(pred.tolist())
                    val_loss += loss.item()

                val_loss /= n_val_steps
                print(f"val_loss {val_loss:.4f} ")
                val_fscores = []
                for lang in val_y_true.keys():
                    val_fscore = f1_score(val_y_true[lang], val_y_pred[lang],
                                          average="macro")
                    print(f"\t{lang} val_fscore {val_fscore:.3f}")
                    val_fscores.append(val_fscore)
                if len(list(val_y_true.keys())) > 1:
                    print(f"\tavg val_fscore: {np.mean(val_fscores):.3f}")
                model.train()

    # Evaluation
    model.eval()
    val_y_true = defaultdict(list)
    val_y_pred = defaultdict(list)
    val_loss = 0.0
    for batch in val_loader:
        inputs = {'input_ids':      batch[0].cuda(),
                  'attention_mask': batch[1].cuda(),
                  'token_type_ids': batch[2].cuda(),
                  'labels':         batch[3].cuda()}
        langs = batch[4]

        with torch.no_grad():
            logits = model(**inputs)
            loss = criterion(logits, inputs["labels"])

        for target, pred, lang in zip(inputs["labels"].cpu(),
                                      logits.cpu() >= 0,
                                      langs):
            val_y_true[lang].append(target.tolist())
            val_y_pred[lang].append(pred.tolist())
        val_loss += loss.item()

    val_loss /= n_val_steps
    print(f"val_loss {val_loss:.4f} ")
    val_fscores = []
    for lang in val_y_true.keys():
        val_fscore = f1_score(val_y_true[lang], val_y_pred[lang],
                              average="macro")
        print(f"\t{lang} val_fscore {val_fscore:.3f}")
        val_fscores.append(val_fscore)
    if len(list(val_y_true.keys())) > 1:
        print(f"\tavg val_fscore: {np.mean(val_fscores):.3f}")

    # Test set
    test_y_true = defaultdict(list)
    test_y_pred = defaultdict(list)
    test_loss = 0.0
    n_test_steps = len(test_loader)
    for batch in test_loader:
        inputs = {'input_ids':      batch[0].cuda(),
                  'attention_mask': batch[1].cuda(),
                  'token_type_ids': batch[2].cuda(),
                  'labels':         batch[3].cuda()}
        langs = batch[4]

        with torch.no_grad():
            logits = model(**inputs)
            loss = criterion(logits, inputs["labels"])

        for target, pred, lang in zip(inputs["labels"].cpu(),
                                      logits.cpu() >= 0,
                                      langs):
            test_y_true[lang].append(target.tolist())
            test_y_pred[lang].append(pred.tolist())
        test_loss += loss.item()

    test_loss /= n_test_steps
    print(f"\ttest_loss {test_loss:.3f} ")
    test_fscores = []
    for lang in test_y_true.keys():
        test_fscore = f1_score(test_y_true[lang], test_y_pred[lang],
                               average="macro")
        print(f"\t{lang} test_fscore {test_fscore:.3f}")
        report = classification_report(
            test_y_true[lang], test_y_pred[lang], digits=3,
            target_names=target_encoder.get_feature_names()
        )
        for line in report.split("\n"):
            print("\t", line)

        test_fscores.append(test_fscore)

    if len(list(test_y_true.keys())) > 1:
        print(f"\tavg test_fscore: {np.mean(test_fscores):.3f}")

    if args.save:
        torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-T", "--train_languages", nargs="+", required=True)
    parser.add_argument("-V", "--val_languages", nargs="+", required=True)

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after "
                             "tokenization. Sequences longer than this will "
                             "be truncated, sequences shorter will be padded.")
    parser.add_argument("-b", "--batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("-w", "--weight_decay", default=1e-2, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("-e", "--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("-n", "--val_every", default=99999999, type=int)
    parser.add_argument("-l", "--limit", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", default="O1",
                        choices=["O0", "01", "02", "03"])
    parser.add_argument("-s", "--save")

    args = parser.parse_args()

    assert args.train_languages and args.val_languages

    main(args)
