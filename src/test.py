import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--bert_model", default='bert-base', type=str)

args = parser.parse_args()

print(args.bert_model)
