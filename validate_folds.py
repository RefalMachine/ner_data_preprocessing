from utils.folds_generator import FoldsGenerator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds_masked_dir')
    parser.add_argument('--folds_count', default=4, type=int)
    args = parser.parse_args()

    folds_generator = FoldsGenerator()
    folds_generator.validate_folds(args.folds_masked_dir, args.folds_count)
