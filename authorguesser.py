"""
This script is used to guess a given text based on trained models.
It is fully automated and it guesses all TXT files that are stored in a given directory.
"""

import os
import requests
import joblib
import re
from datetime import datetime
import time
import json
import argparse
import chardet

version = '2.0.1'
authors = ['František Válek', 'Jan Hajič', 'Jiří Szromek', 'Martin Holub' 'Zdenko Vozár']
project_name = 'DigiLab NK ČR'
project_web = 'https://digilab.nkp.cz/'


class TimingLogger:

    def __init__(self):
        self.total_time = 0
        self.preprocessing_time = 0
        self.external_preprocessing_time = 0
        self.prediction_time = 0

        self._separator = ','

    def __str__(self):
        return self._separator.join(
            [str(self.total_time),
             str(self.preprocessing_time),
             str(self.external_preprocessing_time),
             str(self.prediction_time)])


_TIMING = TimingLogger()


author_to_id = {'A. Stašek': 'a-01',
                'J. Neruda': 'a-02',
                'J. Arbes': 'a-03',
                'K. Klostermann': 'a-04',
                'F. X. Šalda': 'a-05',
                'T. G. Masaryk': 'a-06',
                'A. Jirásek': 'a-07',
                'Č. Slepánek': 'a-08',
                'E. Krásnohorská': 'a-09',
                'F. Herites': 'a-10',
                'I. Olbracht': 'a-11',
                'J. Vrchlický': 'a-12',
                'J. S. Machar': 'a-13',
                'J. Zeyer': 'a-14',
                'K. Čapek': 'a-15',
                'K. Nový': 'a-16',
                'K. Sabina': 'a-17',
                'K. V. Rais': 'a-18',
                'K. Světlá': 'a-19',
                'S. K. Neumann': 'a-20',
                'V. Hálek': 'a-21',
                'V. Vančura': 'a-22',
                'Z. Winter': 'a-23'}

id_to_author = {'a-01': 'A. Stašek', 
                'a-02': 'J. Neruda', 
                'a-03': 'J. Arbes', 
                'a-04': 'K. Klostermann', 
                'a-05': 'F. X. Šalda', 
                'a-06': 'T. G. Masaryk', 
                'a-07': 'A. Jirásek', 
                'a-08': 'Č. Slepánek', 
                'a-09': 'E. Krásnohorská', 
                'a-10': 'F. Herites', 
                'a-11': 'I. Olbracht', 
                'a-12': 'J. Vrchlický', 
                'a-13': 'J. S. Machar', 
                'a-14': 'J. Zeyer', 
                'a-15': 'K. Čapek', 
                'a-16': 'K. Nový', 
                'a-17': 'K. Sabina', 
                'a-18': 'K. V. Rais', 
                'a-19': 'K. Světlá', 
                'a-20': 'S. K. Neumann', 
                'a-21': 'V. Hálek', 
                'a-22': 'V. Vančura', 
                'a-23': 'Z. Winter'}

""" ARGUMENTS ------------------------------------------------------------------- """
def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    """ Models and data paths """
    parser.add_argument('-m', '--models_path', action='store', required=False,
                        help='Path to directory where models are be stored.')

    parser.add_argument('-i', '--input_path', action='store', required=False,
                        help='Path to directory where TXT files to be guessed are stored.')

    parser.add_argument('-o', '--output_path', action='store', required=False,
                        help='Path to directory where guessed data should be stored.')
    
    parser.add_argument('-d', '--delete_input', action='store', required=False,
                        help='Should input be deleted? y/n')

    parser.add_argument('-l', '--log', action='store', required=False,
                        help='Log performance information into a CSV file. The columns'
                             ' are: total time, time spent in preprocessing, time spent'
                             ' in external services (UDPipe) during preprocessing, and'
                             ' time spent in the classification model\'s prediction method.')

    return parser

""" PROCESSING INPUT TEXT ------------------------------------------------------- """


def clear_document(input_string:str):
    """ This function provides a basic cleaning of the input file (from headers, footers, page numbers, footnotes etc.) """
    # create list of num chars, because evaluating from string causes more problems.
    num_chars = '1234567890'

    page_lines = input_string.split('\n')

    cleared_lines = []

    footnotes_were_cut = False
    for i, line in enumerate(page_lines):
        try:
            if i == 0 or i == 1:
                if len(line) <= 25:
                    # NOTE: u těch druhých řádek to může být problematické!
                    continue
                elif line.isupper():
                    continue
                elif line.count(' ') >= len(line)/3:
                    continue
                else:
                    cleared_lines.append(line)
            elif i == len(page_lines)-1:                         
                if len(line) <= 15:
                    if line[-1] in ['.', '?', '!', '"', ':', ';', '«', '-', '—']:
                        cleared_lines.append(line)
                    else:
                        continue
                elif line.isupper():
                    continue
                elif line.count(' ') >= len(line)/2:
                    continue
                elif line[1] == ')' or line[2] == ')' or line[3] == ')':
                    if line[0] in num_chars:
                        footnotes_were_cut = True
                        break
                    elif line[0] == '*':
                        footnotes_were_cut = True
                        break
                    else:
                        cleared_lines.append(line)
                elif line[0] in num_chars or line[-1] in num_chars:
                    # NOTE: If the last line starts or ends with a number, it is probably footer.
                    continue                   
                else:
                    cleared_lines.append(line)
            elif line[1] == ')' or line[2] == ')' or line[3] == ')':
                if line[0] in num_chars:
                    footnotes_were_cut = True
                    break                     
                elif line[0] == '*':
                    footnotes_were_cut = True
                    break
                else:
                    cleared_lines.append(line)
            else:
                cleared_lines.append(line)
        except IndexError:
            cleared_lines.append(line)

            if footnotes_were_cut:
                # NOTE: If footnotes were cut of, the last line before them can be wrong?
                # NOTE: This occured in one case... there is a possibility that it will cause more problems than it solves...
                last_line = cleared_lines[-1]
                if len(last_line) <= 15:
                    if line[-1] in ['.', '?', '!', '"', ':', ';', '«', '-', '—']:
                        continue
                    else:
                        cleared_lines.pop(-1)

    cleared_text = '\n'.join(cleared_lines)
    return cleared_text


def concat_str(text: str, chars: str) -> str:
    return ''.join(text.split(chars))


def connect_lines(input_string:str):
    """ This function connects the input string into one line, connecting hyphenated words. """
    connected_string = concat_str(input_string, '-\n')
    connected_string = concat_str(connected_string, '—\n')
    connected_string = ' '.join(connected_string.split('\n'))

    return connected_string


""" ENRICHMENT ----------------------------------------------------------------------------- """


def udpipe_text(input_text:str):
    """ This function serves to retrieve UDPiped text. """
    print('\tParsing data through UDPipe...')

    input_data = {
        'tokenizer': '',
        'tagger': '',
        'parser': '',
        'data': input_text
    }

    _udpipe_start_time = time.time_ns()  # TIMING

    response = requests.post('http://lindat.mff.cuni.cz/services/udpipe/api/process', input_data)

    data = eval(response.text)

    _udpipe_end_time = time.time_ns()  # TIMING
    _TIMING.external_preprocessing_time += _udpipe_end_time - _udpipe_start_time  # TIMING

    return data["result"]


def transform_udipiped_to_xml(input_udipiped_text:str):
    """ This function transforms UDPiped data into XML(TEI)-like format for further editing. """
    udpiped_lines = input_udipiped_text.split('\n')
    
    output_data = ''

    for line in udpiped_lines:
        if line == '':
            continue
        elif line[0] == '':
            continue
        elif line[0] == '#':
            if line[:6] == '# sent':
                if output_data == '':
                    output_data = f'{output_data}<s>'
                else:
                    output_data = f'{output_data}\n</s>\n<s>'
            else:
                continue
        else:
            line_data = line.split('\t')
            n = line_data[0]
            token = line_data[1]
            lemma = line_data[2]
            pos = line_data[3]
            morf_tag = line_data[4]
            msd = line_data[5]
            
            if pos == 'PUNCT':
                # TODO: Měli bychom sledovat i join, ale zdálo se mi, že tam něco blbne z UDPipe a nevidím v tom, kde jsou a kde nejsou mezery
                output_data = f'{output_data}\n\t<pc n="{n}" pos="{pos}" join="??" morph="{morf_tag}" msd="" lemma="{lemma}">{token}</pc>'
            else:
                output_data = f'{output_data}\n\t<w n="{n}" pos="{pos}" morph="{morf_tag}" msd="{msd}" lemma="{lemma}">{token}</w>'

    # Adding final sentence closure.
    output_data = f'{output_data}\n</s>'

    # Repairing """ --> '"' so as not to mes up the data
    output_data = output_data.replace('"""', "'\"'")
    
    return output_data


def get_line_word(line_data:str):
    """ This function gets word (or punctuuation sign) out of the UDPiped XML file. """
    if '</pc>' == line_data[-5:]:
        line_type = 'pc'
    else:
        line_type = 'w'

    word_data_1 = line_data.replace(f'</{line_type}>', '')
    if word_data_1[-1] == '>':
        word = '>'
    else:
        word_data = word_data_1.split('>')
        word = word_data[-1]

    return word


def get_line_data(line_data:str):
    """ This function returns UDPiped data from line of XML file """
    if '</pc>' == line_data[-5:]:
        line_type = 'pc'
    else:
        line_type = 'w'

    word_data_1 = line_data.replace(f'</{line_type}>', '')
    if word_data_1[-1] == '>':
        word = '>'
    else:
        word_data = word_data_1.split('>')
        word = word_data[-1]

    metadata_1 = line_data.replace(f'>{word}</{line_type}>', '')
    metadata_1 = metadata_1.replace(f'<{line_type} ', '')
    while '\t' in metadata_1:
        metadata_1 = metadata_1.replace('\t', '')

    metadata_parts = metadata_1.split(' ')

    # NOTE: following metadata are in format "n="1" and not "1""
    if line_type == 'w':
        n = metadata_parts[0]
        pos = metadata_parts[1]
        morph = metadata_parts[2]
        msd =metadata_parts[3]
        lemma = metadata_parts[4]
      
    elif line_type == 'pc':
        n = metadata_parts[0]
        pos = metadata_parts[1]
        morph = metadata_parts[3]
        msd =metadata_parts[4]
        lemma = metadata_parts[5]

    return line_type, n, pos, morph, msd, lemma, word


def process_input_file_with_UDPipe(input_filename:str):
    """ This function combines all of the functions above and process the input file into a XML-like string combining enriched data by both UDPipe and NameTag. """
    try:
        with open(os.path.join(TEXT_TO_GUESS_PATH, input_filename), 'r', encoding='utf-8') as input_file:
            input_string = input_file.read()
    except UnicodeDecodeError:
        with open(os.path.join(TEXT_TO_GUESS_PATH, input_filename), 'rb') as source_file:
            raw_data = source_file.read()
            encoding_result = chardet.detect(raw_data)
            charenc = encoding_result['encoding']  
            
        with open(os.path.join(TEXT_TO_GUESS_PATH, input_filename), 'r', encoding=charenc) as input_file:
            input_string = input_file.read()

    cleared_input = clear_document(input_string=input_string)
    connected_input = connect_lines(input_string=cleared_input)

    UDPipe_enriched_data = udpipe_text(input_text=connected_input)
    UDPipe_XML_data = transform_udipiped_to_xml(input_udipiped_text=UDPipe_enriched_data)

    return UDPipe_XML_data, connected_input


""" DELEXICALIZING DATA ---------------------------------------------------------------------------------------------- """


def get_line_data_of_r_01(line_data:str):
    """ This function returns line data from the UDPiped preprocessed XML data. """
    if '</pc>' == line_data[-5:]:
        line_type = 'pc'
    else:
        line_type = 'w'

    word_data_1 = line_data.replace(f'</{line_type}>', '')
    try:
        if word_data_1[-1] == '>':
            word = '>'
        else:
            word_data = word_data_1.split('>')
            word = word_data[-1]
    except IndexError:
        print(line_data)

    metadata_1 = line_data.replace(f'>{word}</{line_type}>', '')
    metadata_1 = metadata_1.replace(f'<{line_type} ', '')
    while '\t' in metadata_1:
        metadata_1 = metadata_1.replace('\t', '')

    metadata_parts = metadata_1.split(' ')

    n = metadata_parts[0][:-1].replace('n="', '')
    pos = metadata_parts[1][:-1].replace('pos="', '')
    morph = metadata_parts[2][:-1].replace('morph="', '')
    msd =metadata_parts[3][:-1].replace('msd="', '')
    if 'lemma=\'"\'' in metadata_parts[4]:
        lemma = metadata_parts[4][:-1].replace("lemma='", '')
    else:
        lemma = metadata_parts[4][:-1].replace('lemma="', '')

    return n, pos, morph, msd, lemma, word


def delexicalize_XML_data_to_r_08(input_enriched_data:str):
    """ This function delexicalizes data according to r-08 preprocessing code (='auto_pos_lemma': '08', všechny autosémantické tokeny nahrazeny UDPIpe POS tagy, zbytek UDPipe lemmata). """

    autosemantic_pos = ['NOUN', 'ADJ', 'VERB', 'ADV', 'NUM']

    lines_in_input = input_enriched_data.split('\n')

    output_str = ''

    for line_data in lines_in_input:
        if line_data == '':
            continue
        elif '<s>' in line_data or '</s>' in line_data or '</passage' in line_data:
            output_str += f'{line_data}\n'
        elif '<passage' in line_data:
            line_data = line_data.replace('.r-03.', f'.{r}.')
            output_str += f'{line_data}\n'
        else:
            n, pos, morph, msd, lemma, word = get_line_data_of_r_01(line_data)
            if pos in autosemantic_pos:
                output_str += f'\t\t<token>POS_{pos}</token>\n'
            else:
                output_str += f'\t\t<token>{lemma}</token>\n'

    return output_str


def process_input_file(input_filename:str):
    """ This function processes an input file into a XML-like file in r-08 preprocessing. """
    UDPipe_XML_data, connected_input = process_input_file_with_UDPipe(input_filename=input_filename)

    delexicalized_data = delexicalize_XML_data_to_r_08(input_enriched_data=UDPipe_XML_data)

    return delexicalized_data, connected_input


""" GUESSING FUNCTIONS ----------------------------------------------------------------------------------------- """


def get_token_count(XML_data:str):
    """ This function returns numbet of tokens present in the XML data. """
    XML_lines = XML_data.split('\n')

    token_count = 0

    for line in XML_lines:
        if '</token>' in line:
            token_count += 1
    
    return token_count


def get_token_contents(tagged_line:str):
    if '<s>' in tagged_line or '</s>' in tagged_line or '</passage>' in tagged_line:
        return False
    else:
        contents = re.sub('<token>', '', tagged_line)
        contents = re.sub('</token>', '', contents)
        return contents


def get_tokens_from_xml_data(input_xml_data:str):
    """ This function prepares the data in XML fromat into input that is needed for vectorization. """
    xml_lines = input_xml_data.split('\n')

    output_data = ''

    for line in xml_lines:
        token_contents = get_token_contents(line)
        if token_contents:
            output_data += f'{token_contents} '

    return output_data[:-1]


def get_vectorizer_filename_from_model_filename(model_filename:str):
    """ This function gets verctorizer name from the model filename - because vectorizer is the same for all of the models with the same r, s, and n parameters. """
    model_name_parts = model_filename.split('.')
    vectorizer_filename = f'vectorizer_{model_name_parts[0]}.{model_name_parts[1]}.{model_name_parts[2]}.joblib'
    
    return vectorizer_filename


def guess_instance(model_filename:str, data_to_eval:str):
    """
    This function is used within the guess_file function as a guess of one instance.
    
    :param model_filename: quite self-explanatory...
    :param data_to_eval: input data must be a str in XML-like structure, already delexicalized (with "<token>TOKEN</token>")
    """

    model = joblib.load(os.path.join(MODELS_PATH, model_filename))
    vectorizer = joblib.load(os.path.join(MODELS_PATH, get_vectorizer_filename_from_model_filename(model_filename)))

    data_for_vectorizer = get_tokens_from_xml_data(input_xml_data=data_to_eval)

    # Create test counts
    X_guess_counts = vectorizer.transform([data_for_vectorizer])

    # Prediction and evaluation:
    y_guess_pred = model.predict(X_guess_counts)
    
    print(f'\tGUESS: model - {model_filename}, guessed author ID - {y_guess_pred[0]}')

    return y_guess_pred[0]


def select_model_file(list_of_models:list, s:int):
    """ This function selects the best of suggested models according to segment length of the input. """
    list_of_models.sort()
    
    if s >= 990:
        for model in list_of_models:
            if 's-1000.' in model and model.startswith('r-'):
                return model
    elif s >= 490:
        for model in list_of_models:
            if 's-500.' in model and model.startswith('r-'):
                return model
    elif s >= 195:
        for model in list_of_models:
            if 's-200.' in model and model.startswith('r-'):
                return model
    elif s >= 95:
        for model in list_of_models:
            if 's-100.' in model and model.startswith('r-'):
                return model
    else:
        for model in list_of_models:
            if 's-50.' in model and model.startswith('r-'):
                return model


def guess_file(input_filename:str, models_path:str):
    """ This function guesses the file in all of the relevant  """
    _preprocessing_start_time = time.time_ns()  # TIMING

    delexicalized_data, connected_input = process_input_file(input_filename=input_filename)

    _preprocessing_end_time = time.time_ns() # TIMING
    _TIMING.preprocessing_time += _preprocessing_end_time - _preprocessing_start_time # TIMING

    # count of tokens is the same in all of the delexicalization versions, so we need to get the token count only once
    token_count = get_token_count(XML_data=delexicalized_data)

    model_to_run = select_model_file(list_of_models=os.listdir(models_path), s=token_count)

    _prediction_start_time = time.time_ns() # TIMING

    guessed_auhtor_id = guess_instance(model_filename=model_to_run, data_to_eval=delexicalized_data)

    _prediction_end_time = time.time_ns() # TIMING
    _TIMING.prediction_time += _prediction_end_time - _prediction_start_time # TIMING

    guessed_auhtor = id_to_author[guessed_auhtor_id]
    print(f'\tThe guessed author of {input_filename} is: {guessed_auhtor}')
    date_of_guess = datetime.today().strftime("%Y-%m-%d")

    XML_guessed = f'<info>\n\t<guessed_author>{guessed_auhtor}</guessed_author>\n\t<guessed_author_ID>{guessed_auhtor_id}</guessed_author_ID>\n\t<original_filename>{input_filename}</original_filename>\n\t<model_used>{model_to_run}</model_used>\n\t<date>{date_of_guess}</date>\n</info>\n' + delexicalized_data + f'\n<original_string>{connected_input}</original_string>\n'

    JSON_entry = {'guessed_author': guessed_auhtor,
                  'guessed_author_ID': guessed_auhtor_id,
                  'original_filename': input_filename,
                  'model_used': model_to_run,
                  'date': date_of_guess,
                  'associated_XML': ''}

    return guessed_auhtor, XML_guessed, JSON_entry


def guess_all_files(input_path:str, output_path:str,  models_path:str, remove_original=False):
    """ The main function that executes the guess all files in a given path. """
    files_to_guess = os.listdir(input_path)

    files_to_skip = []
    # Test if there are any files over the limit of 18 000 characters (10 standard pages) and texts treir encoding of other that UTF-8
    print('Checking lengths and encodings...')
    for file_to_guess in files_to_guess:
        try:
            with open(os.path.join(input_path, file_to_guess), 'r', encoding='utf-8') as input_file:
                data_in_file = input_file.read()
                len_of_input = len(data_in_file)
                if len_of_input > 18000:
                    print(f'File {file_to_guess} has over 18 000 characters. Input must be below 18 000 characters. It will not be guessed.')
                    files_to_skip.append(file_to_guess)
        except UnicodeDecodeError:
            with open(os.path.join(input_path, file_to_guess), 'rb') as source_file:
                raw_data = source_file.read()
                encoding_result = chardet.detect(raw_data)
                charenc = encoding_result['encoding']  
                
            with open(os.path.join(input_path, file_to_guess), 'r', encoding=charenc) as input_file:
                data_in_file = input_file.read()
                len_of_input = len(data_in_file)
                if len_of_input > 18000:
                    print(f'File {file_to_guess} has over 18 000 characters. Input must be below 18 000 characters. It will not be guessed.')
                    files_to_skip.append(file_to_guess)

    for file_to_guess in files_to_guess:
        if file_to_guess in files_to_skip:
            continue
        else:
            print('GUESSING...', file_to_guess)
            guessed_auhtor, XML_guessed, JSON_entry = guess_file(input_filename=file_to_guess, models_path=models_path)

            # Save the results into a XML file
            guessed_filename = f'guess_uuid_{str(time.time()).replace(".", "")}_{file_to_guess[:-4]}.xml'
            with open(os.path.join(output_path, guessed_filename), 'w', encoding='utf-8') as guessed_file:
                guessed_file.write(XML_guessed)
                print('\tGuessed output saved as', guessed_filename)

            # Save the results into JSON
            JSON_filename = 'results.json'
            JSON_entry['associated_XML'] = guessed_filename

            if not os.path.exists(os.path.join(output_path, JSON_filename)):
                evaluation_id = 0
                with open(os.path.join(output_path, JSON_filename), 'w', encoding='utf-8') as json_results_file:
                    json.dump({'last_run_id': evaluation_id, evaluation_id: JSON_entry}, json_results_file)
            else:
                with open(os.path.join(output_path, JSON_filename), 'r', encoding='utf-8') as json_results_file:
                    json_data = json.load(json_results_file)
                
                last_run_id = json_data['last_run_id']
                current_run_id = last_run_id+1
                json_data['last_run_id'] = current_run_id
                json_data[current_run_id] = JSON_entry
            
                with open(os.path.join(output_path, JSON_filename), 'w', encoding='utf-8') as json_results_file:
                    json.dump(json_data, json_results_file, indent=4)  

            if remove_original:
                os.remove(os.path.join(input_path, file_to_guess))
                print(f'\tOriginal file {file_to_guess} has been removed.')


if __name__ == '__main__':
    _script_start = time.time()

    parser = build_argument_parser()
    args = parser.parse_args()

    ROOT_PATH = os.getcwd()

    TEXT_TO_GUESS_PATH = os.path.join(ROOT_PATH, 'texts_to_guess')
    MODELS_PATH = os.path.join(ROOT_PATH, 'models')
    GUESSED_PATH = os.path.join(ROOT_PATH, 'guessed_files')

    remove_originals = True

    if args.models_path:
        MODELS_PATH = args.models_path

    if args.input_path:
        TEXT_TO_GUESS_PATH = args.input_path

    if args.output_path:
        GUESSED_PATH = args.output_path

    if args.delete_input:
        if args.delete_input == 'y':
            remove_originals = True
        if args.delete_input == 'n':
            remove_originals = False

    guess_all_files(remove_original=remove_originals, models_path=MODELS_PATH, input_path=TEXT_TO_GUESS_PATH, output_path=GUESSED_PATH)

    _script_end = time.time()
    _TIMING.total_time = _script_end - _script_start

    if args.log:
        LOG_PATH = args.log_path
        with open(LOG_PATH, 'a') as log_fh:
            log_fh.write(str(_TIMING) + '\n')
