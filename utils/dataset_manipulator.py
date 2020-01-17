import os
import codecs
from sklearn import utils
from shutil import copyfile
from utils.data_processor import BertNerFormatProcessor
import logging

logger = logging.getLogger(__name__)

class DatasetManipulator:
    _allowable_types = ['ann_fromat', 'bert']
    def __init__(self, bert_tokenizer_dir):
        self.bertnf_processor = BertNerFormatProcessor(bert_tokenizer_dir)

    def get_docs_from_ann_format(self, data_dir):
        # create .bertnf files
        self.bertnf_processor.process_dir_and_write(data_dir)
        # get docs
        data_docs = []
        for file_name in os.listdir(data_dir):
            if not file_name.endswith('.bertnf'):
                continue
            file_path = os.path.join(data_dir, file_name)

            with codecs.open(file_path, 'r', 'utf-8') as file:
                data = file.read().strip()
                data_docs.append(data)

        data_docs = '\n\n'.join(data_docs).split('<DOCSTART>\n\n')
        data_docs = [data.strip() for data in data_docs if '\t' in data]

        return data_docs

    def get_docs_from_bert_format(self, data_dir):
        train_file_path = os.path.join(data_dir, 'train.txt')
        with codecs.open(train_file_path, 'r', 'utf-8') as file:
            train_data = file.read().strip()

        train_docs = train_data.split('<DOCSTART>\n\n')
        train_docs = [data.strip() for data in train_docs if '\t' in data]

        return train_docs

    def create_dataset(self, rules):
        """rules = {output: dir_out,
                    input: [(dir_in_1, type), (dir_in_2, type)...],
                    dev_path: dev_path,
                    test_path: test_path}
           type = ann_format | bert"""
        total_train_docs = []
        for data_dir, data_type in rules['input']:
            if data_type == 'ann_format':
                docs = self.get_docs_from_ann_format(data_dir)
            elif data_type == 'bert':
                docs = self.get_docs_from_bert_format(data_dir)
            else:
                raise Exception(f'Unrecognized type {data_type}. Please use one from {self._allowable_types}')
            logger.info(f'Extracted {len(docs)} documents from {data_dir} with type {data_type}')
            total_train_docs += docs

        output_dir = rules['output']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        train_data_path = os.path.join(output_dir, 'train.txt')
        total_train_docs = utils.shuffle(total_train_docs)
        total_train_docs = '<DOCSTART>\n\n' + '\n\n<DOCSTART>\n\n'.join(total_train_docs)
        with codecs.open(train_data_path, 'w', 'utf-8') as file:
            file.write(total_train_docs)

        if rules.get('test_path', None) is not None:
            src_test_data_path = rules['test_path']
            test_data_path = os.path.join(output_dir, 'test.txt')
            copyfile(src_test_data_path, test_data_path)

        if rules.get('dev_path', None) is not None:
            src_dev_data_path = rules['dev_path']
            dev_data_path = os.path.join(output_dir, 'dev.txt')
            copyfile(src_dev_data_path, dev_data_path)

        logger.info(f'Created dataset in {output_dir}')
