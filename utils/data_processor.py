import os
import codecs
import re
import nltk
from transformers import BertTokenizer
from tqdm import tqdm


class SentenceTokenizer():
    """Sentence tokenizer, with respect of ann data. One entity can't be in several sentences.
       Also recalculate ann information for sentences.
    """
    def __init__(self):
        pass

    def tokenize(self, text):
        text_sentences = []
        for paragraph in text.replace('\r\n', '\n').split('\n'):
            if len(paragraph) == 0:
                continue
            sentences = nltk.sent_tokenize(paragraph, language="russian")
            text_sentences += sentences

        return text_sentences

    def tokenize_with_ann(self, txt_path, ann_path, return_shift_list=False, test=True):
        """Main method. Takes paths for txt ann pair and tokenize text and ann on sentences.
        """
        with codecs.open(txt_path, 'r', 'utf-8') as file:
            text_data = file.read().rstrip()

        with codecs.open(ann_path, 'r', 'utf-8') as file:
            ner_data_list = file.read().split('\n')

        ner_data_list = [data.split('\t')[1].split()[:2] + data.split('\t')[1].split()[-1:] + [data.split('\t')[2]] for data in ner_data_list if len(data) > 0]
        ner_data_list = list(map(lambda x: (x[0], int(x[1]), int(x[2]), x[3]), ner_data_list))
        ner_data_list = sorted(ner_data_list, key=lambda x: x[1])

        text_sentences = self.tokenize(text_data)
        shift_list = self.calculate_shifts(text_data, text_sentences)

        assert len(shift_list) == len(text_sentences)
        assert len(text_data) == sum([len(sent) for sent in text_sentences]) + sum(shift_list)
        text_sentences, ner_data_list_by_sent, shift_list = self.fix_sentences_and_split_ner(text_sentences, ner_data_list, shift_list)

        assert len(text_sentences) == len(ner_data_list_by_sent)
        if test is True:
            ##
            #test thats all OK
            replaced_full = self.replace_ner_in_text_to_tag(text_data, ner_data_list)
            to_spaces_re = re.compile('[ \n\r\t]+', re.DOTALL)
            replaced_full = re.sub(to_spaces_re, ' ', replaced_full).strip()
            replaced_by_sent_list = []
            for i in range(len(text_sentences)):
                sent_text_data = text_sentences[i]
                sent_ner_data_list = ner_data_list_by_sent[i]
                replaced_by_sent_list.append(self.replace_ner_in_text_to_tag(sent_text_data, sent_ner_data_list))

            assert len(replaced_by_sent_list) == len(shift_list)

            replaced_by_sent = replaced_by_sent_list[0]
            for i in range(len(replaced_by_sent_list)):
                if i == 0:
                    continue
                replaced_by_sent += ' ' * shift_list[i] + replaced_by_sent_list[i]

            replaced_by_sent = re.sub(to_spaces_re, ' ', replaced_by_sent).strip()
            assert replaced_full == replaced_by_sent
            ##

        output = (text_sentences, ner_data_list_by_sent)
        if return_shift_list is True:
            output = output + (shift_list,)
        return output

    def calculate_shifts(self, text, sentences):
        shift_list = []
        last_sent_end_pos = 0
        for i in range(0, len(sentences)):
            sent_start_pos = text.find(sentences[i], last_sent_end_pos)

            assert sent_start_pos >= 0

            shift_list.append(sent_start_pos - last_sent_end_pos)
            last_sent_end_pos = sent_start_pos + len(sentences[i])

        return shift_list

    def fix_sentences_and_split_ner(self, sentences, ner_data_list, shift_list):
        fixed_sentences = [sentences[0]]
        fixed_shift_list = [shift_list[0]]
        ner_data_list_by_sent = []
        shift = 0
        # merge sent to prev if sentence to short or ner separeted by two sentences
        for i, sent in enumerate(sentences):
            shift += shift_list[i]
            if i > 0:
                if not self.check_ner_boundary(shift, ner_data_list) or len(sent.split()) <= 3:
                    fixed_sentences[-1] += ' ' * shift_list[i] + sent
                else:
                    fixed_sentences.append(sent)
                    fixed_shift_list.append(shift_list[i])

            shift += len(sent)

        ner_data_list_by_sent = []
        shift = 0
        for i, sent in enumerate(fixed_sentences):
            shift += fixed_shift_list[i]
            sent_ner_data_list = [ner_data for ner_data in ner_data_list if ner_data[1] >= shift and ner_data[1] < shift + len(sent)]
            sent_ner_data_list = [(ner_data[0], ner_data[1] - shift, ner_data[2] - shift, ner_data[3]) for ner_data in sent_ner_data_list]
            ner_data_list_by_sent.append(sent_ner_data_list)
            shift += len(sent)

        return fixed_sentences, ner_data_list_by_sent, fixed_shift_list

    @staticmethod
    def replace_ner_in_text_to_tag(text, ner_data_list):
        replaced_text = ''
        last_pos = 0
        for ner_data in ner_data_list:
            replaced_text += text[last_pos: ner_data[1]] + ner_data[0]
            last_pos = ner_data[2]
        if last_pos != len(text):
            replaced_text += text[last_pos:]

        return replaced_text

    @staticmethod
    def check_ner_boundary(pos, ner_data_list):
        closes_ner_to_right = None
        for i in range(len(ner_data_list)):
            if ner_data_list[i][2] > pos:
                closes_ner_to_right = ner_data_list[i]
                break

        if closes_ner_to_right is None:
            return True

        return closes_ner_to_right[1] >= pos

class BertNerFormatProcessor():
    """Class for generation bertnf files for txt ann pairs
    """
    def __init__(self, tokenizer_root):
        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_root, do_lower_case=False)
        self.sentence_tokenizer = SentenceTokenizer()

    def process_file(self, txt_path, ann_path):
        text_sentences, ner_data_list_by_sent = self.sentence_tokenizer.tokenize_with_ann(txt_path, ann_path)
        tokens_with_tags_by_sent = []
        for i in range(len(text_sentences)):
            tokenized_text, tags = self.prepare_data(text_sentences[i], ner_data_list_by_sent[i])
            assert len(tokenized_text) == len(tags)

            tokens_with_tags = [(tokenized_text[i], tags[i]) for i in range(len(tags))
                                if tokenized_text[i].count('\r') + tokenized_text[i].count('\n') < len(tokenized_text[i])]
            tokens_with_tags_by_sent.append(tokens_with_tags)

        return tokens_with_tags_by_sent

    def process_dir(self, dir_path):
        files_dict = {}
        for file in os.listdir(dir_path):
            filename = os.path.splitext(file)[0]
            if filename not in files_dict:
                files_dict[filename] = {'.ann': None, '.txt': None}
            ext = os.path.splitext(file)[-1]
            if ext != '.ann' and ext != '.txt':
                continue
            files_dict[filename][ext] = os.path.join(dir_path, file)

        files_dict = {file: pair for file, pair in files_dict.items() if len([val for ext, val in pair.items() if val is not None]) == 2}
        process_results = {}
        for filename, pair in tqdm(files_dict.items()):
            try:
                tokens_with_tags_by_sent = self.process_file(pair['.txt'], pair['.ann'])
            except Exception as e:
                print(filename)
                raise e

            doc_text = '\n\n'.join(['\n'.join('\t'.join(line) for line in sent) for sent in tokens_with_tags_by_sent])
            process_results[filename] = doc_text

        return process_results

    def process_dir_and_write(self, dir_path):
        process_results = self.process_dir(dir_path)
        for filename in process_results:
            doc_text = process_results[filename]
            bertnf_path = os.path.join(dir_path, filename + '.bertnf')
            with codecs.open(bertnf_path, 'w', 'utf-8') as out_file_descr:
                out_file_descr.write(f'<DOCSTART>\n\n{doc_text}')

    def prepare_data(self, text_data, ner_data_list):
        start_pos_to_ner_data = {start_byte: (tag, start_byte, end_byte) for tag, start_byte, end_byte, text in ner_data_list}

        try:
            tokenized_text = self.bert_tokenizer.basic_tokenizer.tokenize(text_data)
        except Exception as e:
            print(text_data)
            raise e

        tokens_pos = self.calc_byte_pos_for_tokens(text_data, tokenized_text)

        assert len(tokenized_text) == len(tokens_pos)

        tags = []
        last_ner = None
        for i, (start_byte, end_byte) in enumerate(tokens_pos):
            if last_ner is None:
                if start_byte not in start_pos_to_ner_data:
                    tags.append('O')
                else:
                    last_ner = start_pos_to_ner_data[start_byte]
                    tag = self.replace_tag(last_ner[0])
                    if tag == 'O':
                        tags.append(tag)
                    else:
                        tags.append(f'B-{tag}')
                    last_byte = last_ner[2]
            else:
                if end_byte <= last_byte:
                    tag = self.replace_tag(last_ner[0])
                    if tag == 'O':
                        tags.append(tag)
                    else:
                        tags.append(f'I-{tag}')
                else:
                    tags.append('O')
                    last_ner = None

        return tokenized_text, tags

    @staticmethod
    def calc_byte_pos_for_tokens(text, tokenized_text):
        pos_list = []
        last_pos = 0

        for token in tokenized_text:
            start_byte = text.find(token, last_pos)
            end_byte = start_byte + len(token)
            pos_list.append((start_byte, end_byte))
            last_pos = end_byte

        return pos_list

    @staticmethod
    def replace_tag(tag):
        replaces = {
            'HACKER_GROUP': 'HACKER',
            'MEDIA': 'ORG',
            'GEOPOLIT': 'LOC',
            'POST': 'O',
            'MISC': 'O',
            'ARTEFACT': 'O',
        }
        return replaces.get(tag, tag)
