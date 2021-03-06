from utils.folds_generator import FoldsGenerator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir')
    parser.add_argument('--folds_masked_dir')
    parser.add_argument('--folds_count', default=4, type=int)
    parser.add_argument('--max_diff', default=0.05, type=float)
    parser.add_argument('--dev_size', default=0.2, type=float)

    args = parser.parse_args()

    folds_generator = FoldsGenerator()
    folds_generator.generate_folds(args.dataset_dir, args.folds_masked_dir, args.folds_count, args.max_diff, args.dev_size)
