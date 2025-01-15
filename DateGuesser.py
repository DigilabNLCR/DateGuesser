""" This python script serves to guess the date of the composition based on pre-trained models. """

__version__ = '0.1.0'
__author__  = 'František Válek'

""" This version is a beta version serving to illustrate the guessing for poet Adolf Heyduk. The results are, addmittedly, very poor so far. """

import os
from joblib import load
import plotly.express as px
from collections import defaultdict
import re
import pandas as pd
from colorama import Fore, Style

class NKPdateship:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

        self.target_names = sorted(set(targets))

    def __len__(self):
        return len(self.data)

    def check_valid(self):
        return len(self.data) == len(self.targets)


def extract_token_attributes(line):
    """
    Extracts attributes from a token line.
    :param line: The token line.
    :return: A dictionary with the extracted attributes.
    """
    pattern = r'(\w+)="([^"]*)"'
    attributes = dict(re.findall(pattern, line))
    return attributes

def transform_xml_to_token_data(xml_data, feature_type: str = 'form', autosem_delexicalise: bool = False, ignore_interpunkt: bool = True):
    """
    Transforms tokens in XML data to a given type.
    :param xml_data: The XML data.
    :param feature_type: The feature type to extract (form, lemma, upos, xpos, feats).
    :param delexicalize: Whether to delexicalize the extracted features (autosemantic UPOS tokens are recorded as UPOS in all cases).
    :param ignore_interpunkt: Whether to ignore interpunkt tokens.
    """
    output = ''

    lines = xml_data.split('\n')

    for line in lines:
        if '<token' in line:
            token_attributes = extract_token_attributes(line)
            token_value = token_attributes.get(feature_type, '')
            if autosem_delexicalise and token_attributes['upos'] in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'NUM']:
                token_value = f'UPOS_{token_attributes['upos']}'
            elif ignore_interpunkt and token_attributes['upos'] == 'PUNCT':
                continue
            elif ignore_interpunkt and token_attributes['upos'] == 'SYM':
                continue
            elif ignore_interpunkt and token_attributes['lemma'] in ['.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '"', "'", '„', '“', '”', '–', '—', '…']:
                # NOTE: this has been added because some interpunkt tokens are not marked as PUNCT or SYM (e.g., - or — has been marked as ADP for some reason).
                continue
            output += f'\t\t<token>{token_value}</token>\n'
        else:
            output += line + '\n'

    return output


def split_data_to_statza(xml_data:str):
    """
    Split the XML data into stanzas.
    :param xml_data: The XML data.
    """
    return xml_data.split('</stanza>')[:-1]

def get_tokens_from_stanza(stanza:str):
    """
    Get the tokens from a stanza.
    :param stanza: The stanza.
    """
    tokens = re.findall(r'<token>(.*?)</token>', stanza)
    return tokens

def iterate_over_stanzas(stanzas:list, window_size:int = 2, shift_size:int = 1):
    """
    Iterate over stanzas in a given window size.
    :param stanzas: The stanzas.
    :param window_size: The window size.
    """
    if shift_size > window_size:
        raise ValueError('Shift size must be less than or equal to window size.')
    for i in range(0, len(stanzas) - window_size + 1, shift_size):
        window = stanzas[i:i + window_size]
        yield window

def segment_file(file_path:str, segment_window:int = 1, shift_size:int = 1, feature_type: str = 'form', autosem_delexicalise: bool = False, ignore_interpunkt: bool = True):
    """
    Segment the file into segments of a given size.
    :param file_path: The path to the file.
    :param segment_size: The size of the segments.
    :param feature_type: The feature type to extract (form, lemma, upos, xpos, feats).
    :param autosem_delexicalise: Whether to delexicalize the extracted features (autosemantic UPOS tokens are recorded as UPOS in all cases).
    """
    with open(file_path, 'r', encoding='utf-8') as file_:
        data = file_.read()
        
        transfromed_data = transform_xml_to_token_data(data, feature_type=feature_type, autosem_delexicalise=autosem_delexicalise, ignore_interpunkt=ignore_interpunkt)

        single_stanzas = split_data_to_statza(transfromed_data)
        stanzas_tokens = [get_tokens_from_stanza(stanza) for stanza in single_stanzas]

        segments = iterate_over_stanzas(stanzas_tokens, window_size=segment_window, shift_size=shift_size)

        joined_segments = []
        for segment in segments:
            joined_segment = []
            for stanza in segment:
                joined_segment.extend(stanza)
            joined_segments.append(joined_segment)
    
    return joined_segments


def assigns_year(year: int, window_size: int = 5):
    """
    Assigns a year to a window.
    :param year: The year to assign.
    :param window_size: The size of the window.
    :return: The window label.
    """
    window_start = year - (year % window_size)
    return f'{window_start}-{window_start + (window_size - 1)}'


def count_token(file_path:str):
    """
    Count the tokens in the given file.
    :param file_path: The path to the file.
    """
    # TODO: add ignore interpunkt tokens??

    with open(file_path, 'r', encoding='utf-8') as file_:
        data = file_.read()
        token_count = data.count('<token>')
    
    return token_count


def count_stanza(file_path:str):
    """
    Count the stanzas in the given file.
    :param file_path: The path to the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file_:
        data = file_.read()
        stanza_count = data.count('</stanza>')
    
    return stanza_count


def count_stanza_lens(file_path:str):
    """
    Count the stanzas in the given file.
    :param file_path: The path to the file.
    """

    # TODO: add ignore interpunkt tokens??

    stanza_lens = []

    with open(file_path, 'r', encoding='utf-8') as file_:
        data = file_.read()

        stanzas = data.split('</stanza>')
        for stanza in stanzas:
            token_count = stanza.count('<token>')

            stanza_lens.append(token_count)
    
    return stanza_lens


def join_one_years_period(year:str):
    years_range = year.split('-')
    if years_range[0] == years_range[1]:
        return years_range[0]
    else:
        return year


def count_years_et_tokens(path:str, window_size:int = 10):
    """
    Count the years and tokens in the given path.
    :param path: The path to the folder, where XML files are stored.
    """
    years = defaultdict(int)

    for file_name in os.listdir(path):
        if file_name.endswith('.xml'):
            path_to_file = os.path.join(path, file_name)
            year = int(file_name.split('_')[0])
            period = assigns_year(year, window_size=window_size)
            period = join_one_years_period(period)
            token_count = count_token(path_to_file)
            years[period] += token_count
    
    return years


def count_years_et_stanzas(path:str, window_size:int = 10):
    """
    Count the years and tokens in the given path.
    :param path: The path to the folder, where XML files are stored.
    """
    years = defaultdict(int)

    for file_name in os.listdir(path):
        if file_name.endswith('.xml'):
            path_to_file = os.path.join(path, file_name)
            year = int(file_name.split('_')[0])
            period = assigns_year(year, window_size=window_size)
            period = join_one_years_period(period)
            stanza_count = count_stanza(path_to_file)
            years[period] += stanza_count
    
    return years


def extract_value(text, parameter):
    """
    Extracts the value for a given parameter from the text.
    
    :param text: The text to extract from.
    :param parameter: The parameter to search for.
    :return: The extracted value.
    """
    pattern = rf'{parameter}="([^"]*)"'
    match = re.search(pattern, text)
    return eval(match.group(1)) if match else None


def get_year_et_types_from_composition(file_path:str) -> tuple:
    """
    Gets the year and types from the composition name.
    
    :param composition_name: The name of the composition.
    """

    with open(file_path, 'r', encoding='utf-8') as file_:
        data = file_.read()
        lines = data.split('\n')
        comp_info = lines[-1]

        year = extract_value(comp_info, 'year')
        types = extract_value(comp_info, 'types')

    return year, types


def segment_file_to_data_et_targets(file_path:str, year_window_size=10, segment_window=2, shift_size=1, feature_type='lemma', autosem_delexicalise=True, ignore_interpunkt=True):
    """
    This function creates segments from the given file.
    """
    targets = []
    data = []

    composition_year = get_year_et_types_from_composition(file_path)[0]
    composition_period = assigns_year(composition_year, window_size=year_window_size)

    # NOTE: if the segment_window is greater than number of stanzas in the composition, the window will be set to the number of stanzas.
    # TODO: Evaluate if this is the best approach.
    stanza_count = count_stanza(file_path)
    if segment_window > stanza_count:
        segment_window = stanza_count
        if shift_size > segment_window:
            shift_size = segment_window

    segments = segment_file(file_path=file_path, segment_window=segment_window, shift_size=shift_size, feature_type=feature_type, autosem_delexicalise=autosem_delexicalise, ignore_interpunkt=ignore_interpunkt)

    for seg in segments:
        targets.append(composition_period)

        seg = ' '.join(seg)
        data.append(seg)

    # print('Number of segments:', len(data))
    # print('Number of targets:', len(targets))

    return data, targets


def get_composition_et_vectorizer_from_model_name(model_name:str):
    """
    Get the composition and vectorizer from the model name.

    :param model_name: The model name.
    """
    end_comp = model_name.find('_model')
    composition = model_name[4:end_comp]
    vectorizer = model_name.replace('_model', '_vectorizer')
    end_vect = vectorizer.find('.m-')
    vectorizer = vectorizer[:end_vect]+'.joblib'
    
    return composition, vectorizer


def guess_instance_segment(model_filename:str, vectorizer_filename:str, data_to_eval:str, models_path:str):
    """
    This function is used within the guess_file function as a guess of one instance.
    
    :param model_filename: quite self-explanatory...
    :param data_to_eval: input data must be a str in XML-like structure, already delexicalized (with "<token>TOKEN</token>")
    """

    model = load(os.path.join(models_path, model_filename))
    vectorizer = load(os.path.join(models_path, vectorizer_filename))

    # Create test counts
    X_guess_counts = vectorizer.transform([data_to_eval])

    # Prediction and evaluation:
    y_guess_pred = model.predict(X_guess_counts)
    
    # print(f'\tGUESS: model - {model_filename}, guessed time period - {y_guess_pred[0]}')

    return y_guess_pred[0]


def evaluate_model_on_loo_compostition(model_name:str, models_path:str, documents_path:str):
    """
    This function evaluates the model on the leave-one-out composition.

    :param model_name: The name of the model file (in joblib format).
    :param models_path: The path to the models.
    """
    composition, vectorizer = get_composition_et_vectorizer_from_model_name(model_name)

    print(Fore.BLUE + 'Evaluating model on', composition)
    print(Style.RESET_ALL)

    # TODO: implement the r-designations for the trainnig and testing datasets. Show this on the models filenames, etc. Also, show in the models filenames the window size, segment_window, and shift_size.
    data, targets = segment_file_to_data_et_targets(os.path.join(documents_path, composition), year_window_size=10, segment_window=10, shift_size=5, feature_type='lemma', autosem_delexicalise=True, ignore_interpunkt=True)

    Hejduk_test_dateship = NKPdateship(data, targets)

    score = {'correct': 0, 'incorrect': 0}
    detail = defaultdict(int)

    for i, test_data in enumerate(Hejduk_test_dateship.data):
        true_period = Hejduk_test_dateship.targets[i]

        guessed_period = guess_instance_segment(model_name, vectorizer, test_data, models_path=models_path)

        if guessed_period == true_period:
            score['correct'] += 1
        else:
            score['incorrect'] += 1

        detail[(true_period, guessed_period)] += 1

    eval_score = score['correct'] / (score['correct'] + score['incorrect'])
    print (Fore.GREEN + '\t', composition, 'Score:', score, '\n\t', 'Evaluation score:', eval_score)
    print(Style.RESET_ALL)
    
    return score, eval_score, detail


def transform_detail_to_plotly_heatmap(detail:dict, data_path:str):
    """
    This function transforms the detail dictionary to the plotly heatmap format.
    """

    # NOTE: as we are working now with heyduk only, we can set the periods to the Heyduk's timeframe.
    # TODO: make this more general, so that it can be used for any period (it is, when running on all test documents).
    all_periods = ['1850-1859', '1860-1869', '1870-1879', '1880-1889', '1890-1899', '1900-1909', '1910-1919', '1920-1929']
    
    segments_in_period = defaultdict(int)

    for file_name in os.listdir(data_path):
        if file_name.endswith('.xml'):
            data, targets = segment_file_to_data_et_targets(os.path.join(data_path, file_name), year_window_size=10, segment_window=10, shift_size=5, feature_type='lemma', autosem_delexicalise=True, ignore_interpunkt=True)

            segments_in_period[targets[0]] += len(data)

    data_for_heatmap = []
    for timeframe in all_periods:
        timeframe_line = []
        for timeframe_ in all_periods:
            if (timeframe, timeframe_) not in detail:
                timeframe_line.append(0)
            else:
                timeframe_line.append(detail[(timeframe, timeframe_)]/segments_in_period[timeframe])
        data_for_heatmap.append(timeframe_line)

    df = pd.DataFrame(data_for_heatmap, columns=all_periods, index=all_periods)
    return df

if __name__ == '__main__':
    ROOT = os.getcwd()
    # MODELS_PATH = input('Enter the path to the models you want to ecaluate: ')
    MODELS_PATH = os.path.join(ROOT, 'test_models')
    models_files = os.listdir(MODELS_PATH)

    # DATA_PATH = input('Enter the path to the data to guess: ')
    DATA_PATH = os.path.join(ROOT, 'test_documents')

    results = {}
    details_for_heatmap = defaultdict(int)

    for possible_model in models_files:
        if '_model_n-' in possible_model:
            score, eval_score, detail = evaluate_model_on_loo_compostition(possible_model, models_path=MODELS_PATH, documents_path=DATA_PATH)

            results[possible_model] = {'score': score, 'eval_score': eval_score}

            for key in detail:
                details_for_heatmap[key] += detail[key]

    df_heatmap = transform_detail_to_plotly_heatmap(details_for_heatmap, data_path=DATA_PATH)

    fig = px.imshow(df_heatmap, labels=dict(x='Guessed period', y='True period', color='percentage'), color_continuous_scale='Viridis')

    fig.update_layout(title='Prediction heatmap: Adolf Heyduk, test data',
                      xaxis_title='Guessed period',
                      yaxis_title='True period',
                      width=800,
                      height=800)
    
    fig.show()
