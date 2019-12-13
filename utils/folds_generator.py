import codecs
import os
from sklearn.model_selection import train_test_split
import copy


class FoldsGenerator():
    """Class for folds generation with specified patitions and tag balance. Dataset dir must have .bertnf files
    """
    def __init__(self, max_iter_count=10000):
        self.max_iter_count = max_iter_count

    def generate_folds(self, dataset_dir, folds_masked_dir, folds_count=4, max_diff=0.05, dev_size=0.1):
        all_dataset_docs = self._load_dataset(dataset_dir)
        print(f'All doc count = {len(all_dataset_docs)}')

        docs_for_separate = copy.deepcopy(all_dataset_docs)
        numerator = folds_count - 1
        divider = folds_count
        folds = []
        while len(folds) != folds_count:
            max_diff_mult = len(all_dataset_docs) / len(docs_for_separate)
            current_train_test_partition = {'train': {'mean': numerator / divider, 'max_diff': max_diff * max_diff_mult},
                                            'test': {'mean': 1.0 / divider, 'max_diff': max_diff * max_diff_mult}}

            global_tag_info = self._get_tags_counts(self._double_flat_list(docs_for_separate))
            train, test = self._randomize_and_check_partitions_train_test(docs_for_separate, global_tag_info, current_train_test_partition)
            folds.append(test)

            if len(folds) == folds_count - 1:
                folds.append(train)
            docs_for_separate = copy.deepcopy(train)
            numerator -= 1
            divider -= 1

        self._test_folds(folds, len(all_dataset_docs))

        with_dev_set = type(dev_size) == float and dev_size > 0.0
        for i in range(len(folds)):
            data_root = folds_masked_dir.replace('{}', str(i + 1))
            if not os.path.exists(data_root):
                os.mkdir(data_root)

            train_dev = []
            for j in range(len(folds)):
                if j == i:
                    continue
                train_dev += folds[j]

            if with_dev_set:
                max_diff_mult = len(all_dataset_docs) / len(train_dev)
                global_tag_info = self._get_tags_counts(self._double_flat_list(train_dev))
                partitions = {'train': {'mean': 1.0 - dev_size, 'max_diff': max_diff * max_diff_mult},
                              'test': {'mean': dev_size, 'max_diff': max_diff * max_diff_mult}}
                train, dev = self._randomize_and_check_partitions_train_test(train_dev, global_tag_info, partitions)

            test = folds[i]
            self._get_and_save_test_files(test, os.path.join(data_root, 'test_files.list'))

            train_path = os.path.join(data_root, 'train.txt')
            test_path = os.path.join(data_root, 'test.txt')
            if with_dev_set:
                dev_path = os.path.join(data_root, 'dev.txt')
                train_dev_path = os.path.join(data_root, 'train_dev.txt')

                self._save_data_for_bert(dev, dev_path)
                self._save_data_for_bert(train_dev, train_dev_path)
            else:
                train = train_dev
            self._save_data_for_bert(test, test_path)
            self._save_data_for_bert(train, train_path)

    @staticmethod
    def _save_data_for_bert(data, data_path):
        data = [doc[1:] for doc in data]
        with codecs.open(data_path, 'w', 'utf-8') as file_descr:
            for doc in data:
                file_descr.write('<DOCSTART>\n\n')
                for sent in doc:
                    sent = [line for line in sent if len(line.split()) == 2]
                    file_descr.write('\n'.join(sent))
                    file_descr.write('\n\n')

    @staticmethod
    def _get_and_save_test_files(test, file_path):
        test_files = [doc[0][0] for doc in test]
        test_files.sort()

        with codecs.open(file_path, 'w', 'utf-8') as file:
            file.write('\n'.join(test_files))

    @staticmethod
    def _load_dataset(dataset_dir):
        files = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)
                 if not os.path.isdir(os.path.join(dataset_dir, file)) and file.endswith('.bertnf')]

        all_docs = []
        for file in sorted(files):
            with codecs.open(file, 'r', 'utf-8') as file_descr:
                all_docs += [os.path.basename(file) + file_descr.read().replace('<DOCSTART>', '')]

        all_docs = [[[line for line in sent.split('\n')] for sent in doc.split('\n\n') if len(sent) > 0] for doc in all_docs if len(doc) > 0]
        return all_docs

    @staticmethod
    def _double_flat_list(docs):
        def flat_list(docs):
            return [item for sublist in docs for item in sublist]
        return flat_list(flat_list(docs))

    @staticmethod
    def _get_tags_counts(lines):
        tag_dict = {}
        for line in lines:
            if line.endswith('\n'):
                line = line[:-1]
            if len(line.split()) != 2:
                continue

            token, tag = line.split()
            if tag.startswith('B-'):
                if tag not in tag_dict:
                    tag_dict[tag] = 0
                tag_dict[tag] += 1

        return tag_dict

    @staticmethod
    def _calculate_tags_partition(general_tag_dict, tag_dict):
        return {tag: count / general_tag_dict[tag] for tag, count in tag_dict.items()}

    def _randomize_and_check_partitions_train_test(self, docs, global_tag_info, partitions):
        train, test = train_test_split(docs, test_size=partitions['test']['mean'])
        iteration = 0
        while not self._criteria(self._double_flat_list(train), global_tag_info, partitions['train']) or \
              not self._criteria(self._double_flat_list(test), global_tag_info, partitions['test']):
            if iteration == self.max_iter_count:
                raise Exeption(f'Reached max iteration count: {max_iter_count}')
            train, test = train_test_split(docs, test_size=partitions['test']['mean'])
            iteration += 1
        print(f'Divided for {iteration}')
        return train, test

    def _criteria(self, flat_docs, global_tag_info, partition_info):
        data_tag_info = self._get_tags_counts(flat_docs)
        data_tag_part = self._calculate_tags_partition(global_tag_info, data_tag_info)

        if len(set(data_tag_info).intersection(set(global_tag_info))) != len(global_tag_info):
            return False

        for tag, part in data_tag_part.items():
            if part < partition_info['mean'] - partition_info['max_diff']:
                return False
            if part > partition_info['mean'] + partition_info['max_diff']:
                return False

        print(data_tag_part)
        return True

    @staticmethod
    def _test_folds(folds, doc_count):
        folds_docs = []
        for fold in folds:
            fold_docs = set()
            for doc in fold:
                doc_content = '\n\n'.join(['\n'.join(sent) for sent in doc[1:]]) #doc[0] - name of doc
                fold_docs.add(doc_content)
            folds_docs.append(fold_docs)

        all_docs = set()
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                assert len(folds_docs[i].intersection(folds_docs[j])) == 0
            all_docs.update(folds_docs[i])

        assert len(all_docs) == doc_count
