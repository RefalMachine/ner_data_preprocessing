from utils.folds_generator import FoldsGenerator
from utils.data_processor import BertNerFormatProcessor
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir')
    parser.add_argument('--folds_masked_dir')
    parser.add_argument('--tokenizer_dir')
    parser.add_argument('--folds_count', default=4, type=int)
    parser.add_argument('--max_diff', default=0.05, type=float)
    parser.add_argument('--dev_size', default=0.2, type=float)

    args = parser.parse_args()

    dataset_processor = BertNerFormatProcessor(args.tokenizer_dir)
    folds_generator = FoldsGenerator()

    dataset_processor.process_dir_and_write(args.dataset_dir)
    folds_generator.generate_folds(args.dataset_dir, args.folds_masked_dir, args.folds_count, args.max_diff, args.dev_size)

