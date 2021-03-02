#!/usr/bin/env python3
import argparse
import pandas as pd


def main(args):
    df = pd.read_csv(args.fname_in, index_col=None)
    for lang in args.langs:
        if lang not in df.language.values:
            print(f"ERROR: Language {lang} not in file")
            continue
        df[df.language == lang][["text", "emotion"]].to_csv(
            args.fname_out, index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname_in")
    parser.add_argument("fname_out")
    parser.add_argument("langs", nargs="+")
    parser.add_argument("-d", "--dataset",
                        default="/srv/datasets/emotion/emotion")
    main(parser.parse_args())
