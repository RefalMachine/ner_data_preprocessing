import os
import codecs
import argparse
import json
from utils.dataset_manipulator import DatasetManipulator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config')
    parser.add_argument('--tokenizer_dir')
    args = parser.parse_args()

    manipulator = DatasetManipulator(args.tokenizer_dir)
    all_rules = json.load(codecs.open(args.dataset_config, 'r', 'utf-8'))
    for rules in all_rules:
        manipulator.create_dataset(rules)
