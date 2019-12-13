from utils.data_processor import BertNerFormatProcessor
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir')
    parser.add_argument('--tokenizer_dir')
    args = parser.parse_args()

    dataset_processor = BertNerFormatProcessor(args.tokenizer_dir)
    dataset_processor.process_dir_and_write(args.dataset_dir)
