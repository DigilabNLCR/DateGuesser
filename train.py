""" This script trains models for the nkp-authoguesser project. """
# TODO: ... ADD MORE INFO?
import argparse
import logging
import re
import os
from sys import prefix
import numpy as np
import time

from sklearn.feature_extraction.text import CountVectorizer

from joblib import load, dump

__author__ = "František Válek"
__version__ = "1.0.0"

"""
Version description/updates:
0.0.1: Initial setup for few basic models.
"""

# TODO: add note to skip the empty files in training (?) Maybe not necessary.


""" Shared settings: ---------------------------------------------------------------------------------------------- """
segmentations = ['s-1000', 's-500', 's-200', 's-100', 's-50']
# preprocessing_codes = ['r-04', 'r-05', 'r-06', 'r-07', 'r-08', 'r-09', 'r-10', 'r-11', 'r-12', 'r-13']
preprocessing_codes = ['r-08']


""" CLASSES --------------------------------------------------------------------------------------------------------"""
class NKPauthorship:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

        self.target_names = sorted(set(targets))

    def __len__(self):
        return len(self.data)

    def check_valid(self):
        return len(self.data) == len(self.targets)


""" Argument parser function -------------------------------------------------------------------------------------- """


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    """ Models and data paths """
    parser.add_argument('-m', '--models_path', action='store', required=True,
                        help='Path to directory where models should be stored.')

    parser.add_argument('-d', '--data_path', action='store', required=True,
                        help='Path to directory where all the delexicalized data are stored.')

    """ Models setting: """
    parser.add_argument('-clf', '--model_name', action='store', required=True,
                        help='Model name as in the sklearn package, only some are available in this version.')
    parser.add_argument('-n', '--ngram_range', action='store', required=True,
                        help='set ngram range, use both values even if they are the same; e.g., "1,2" or "2,2".')
    parser.add_argument('-c', '--model_configuration', action='store', required=True,
                        help="Write in model configuration just as you would in model settings, but do not use spaces; "
                             "e.g., C=1.0,gamma=0.001,kernel='rfb',seed=42")

    """ Possibility of limiting training by author_ids, book_ids and passage types. """
    parser.add_argument('-a', '--author_ids', action='store', required=False,
                        help='Set list of author ids, including the "a", like ["a-01", "a-02"]')
    parser.add_argument('-b', '--book_ids', action='store', required=False,
                        help='Set list of book ids, including the "b", like ["b-01", "b-02"]')
    parser.add_argument('-p', '--passage_type', action='store', required=False,
                        help='Set type of passages to train on (all, train, devel, test), train is default')

    """ Some other agrparser functions... """
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def decode_model_configuration(config:str):
    """ This function analyses the input model configuration and transfers it to a dictionary. """
    if config == '':
        return {}
    else:
        return dict([e.split('=')[0], eval(e.split('=')[1])] for e in config.split(','))


def decode_ngram_range_config(ngram_range:str):
    """ This function returns the ngram setting as n_min and n_max. """
    range_ = ngram_range.split(',')
    n_min = eval(range_[0])
    n_max = eval(range_[1])
    return n_min, n_max


""" Functions that work with preprocessed data. --------------------------------------------------------------------"""


def get_relevant_files(r: str, s: str, list_of_all_files: list, b_list=None, a_list=None):
    """ This function extracts relevant datasets based on their delex type and segmentation. """
    # NOTE: possibility of choosing only specific books or authors added
    relevant_files = []

    for file_ in list_of_all_files:
        if r + '.' + s in file_:
            relevant_files.append(file_)

    a_filtered_rel_files = []
    if a_list:
        for file_ in relevant_files:
            for a in a_list:
                if a in file_:
                    a_filtered_rel_files.append(file_)
    else:
        a_filtered_rel_files = relevant_files
     
    b_filtered_rel_files = []
    if b_list:
        for file_ in a_filtered_rel_files:
            for b in b_list:
                if b in file_:
                    b_filtered_rel_files.append(file_)
    else:
        b_filtered_rel_files = a_filtered_rel_files

    return b_filtered_rel_files


def extract_author_id(dataset_filename:str):
    a = re.sub('\.b.+', '', dataset_filename)
    return a


def get_token_contents(tagged_line:str):
    if '<s>' in tagged_line or '</s>' in tagged_line or '</passage>' in tagged_line:
        return False
    else:
        contents = re.sub('<token>', '', tagged_line)
        contents = re.sub('</token>', '', contents)
        return contents


def select_passages_bystr(xml_data, pas_type='test'):
    """ This function returns a list of relevant passages. """
    xml_data = re.sub('\t', '', xml_data)
    lines = xml_data.split('\n')

    out_passages = []

    passage_contents = ''

    if pas_type == 'all':
        for line in lines:
            if 'type=' in line:
                if passage_contents:
                    out_passages.append(passage_contents)
                passage_contents = ''
                continue
            else:
                token_contents = get_token_contents(line)
                if token_contents:
                    passage_contents += f'{token_contents} '
    
    else:
        append = False
        for line in lines:
            if f'type="{pas_type}"' in line:
                if passage_contents:
                    out_passages.append(passage_contents)
                passage_contents = ''
                append = True
                continue
            elif 'type=' in line:
                if passage_contents:
                    out_passages.append(passage_contents)
                passage_contents = ''
                append = False
                continue
            else:
                if append:
                    token_contents = get_token_contents(line)
                    if token_contents:
                        passage_contents += f'{token_contents} '

    return out_passages


def nkp_authorship_to_dataset(path_to_data: str, list_of_files: list, pas_type='test') -> NKPauthorship:
    """
    This function prepares relevant passages into NKPauthorship class.
    Data are formatted as Beautiful Soup objects.
    """

    data = []
    targets = []

    for file_ in list_of_files:
        a = extract_author_id(file_)
        with open(os.path.join(path_to_data, file_), 'r', encoding='utf-8') as dataset_f:
            xml_data = dataset_f.read()
            selected_passages = select_passages_bystr(xml_data, pas_type=pas_type)
            for passage in selected_passages:
                data.append(passage)
                targets.append(a)

    nkp_dataset = NKPauthorship(data, targets)

    return nkp_dataset


""" Main training functions. ---------------------------------------------------------------------------------------"""


def select_model(model_name:str):
    """ This function selects relevant model based on input. """
    if model_name == 'MultinomialNB':
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB
    elif model_name == 'SVC':
        from sklearn.svm import SVC
        return SVC
    elif model_name == 'LinearSVC':
        from sklearn.svm import LinearSVC
        return LinearSVC
    elif model_name == 'KNeighborsClassifier':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier
    elif model_name == 'SGDClassifier':
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier
    elif model_name == 'DecisionTreeClassifier':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier
    else:
        print('Selected model is not available in this version')
        return


def train_model(args, a_list=None, b_list=None, pass_type_to_train_on='train'):
    """ This is the main function that trains the model based on some setting provided via arg. """
    n_min, n_max = decode_ngram_range_config(args.ngram_range)
    model_configuration = decode_model_configuration(args.model_configuration)
    model_name = args.model_name

    for r in preprocessing_codes:
        for s in segmentations:
            model_training_start = time.time()
            print('Training', r, s)
            files_to_process = get_relevant_files(r, s, os.listdir(args.data_path), b_list=b_list, a_list=a_list)

            nkp_train_dataset = nkp_authorship_to_dataset(args.data_path, files_to_process, pas_type=pass_type_to_train_on)
            vectorizer = CountVectorizer(ngram_range=(n_min, n_max))

            X_train_counts = vectorizer.fit_transform(nkp_train_dataset.data)

            dump(vectorizer, os.path.join(args.models_path, f'vectorizer_{r}.{s}.n-{n_min},{n_max}.joblib'))

            model = select_model(model_name)

            clf = model(**model_configuration)

            clf.fit(X_train_counts, nkp_train_dataset.targets)

            dump(clf, os.path.join(args.models_path,
                                   f'{r}.{s}.n-{n_min},{n_max}.m-{model_name}.c-{args.model_configuration}.joblib'))
            model_training_end = time.time()
            print('\tmodel trained in', model_training_end-model_training_start, 'seconds')


if __name__ == '__main__':
    start = time.time()
    
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    if args.author_ids:
        a_list = eval(args.author_ids)
    else:
        a_list = None
    
    if args.book_ids:
        b_list = eval(args.book_ids)
    else:
        b_list = None

    if args.passage_type:
        pass_type_to_train_on = args.passage_type
        print(f'training on {pass_type_to_train_on} passages')
    else:
        pass_type_to_train_on = 'train'

    train_model(args, b_list=b_list, a_list=a_list, pass_type_to_train_on=pass_type_to_train_on)

    end = time.time()
    print('Full time needed for training:', end-start, 'seconds')