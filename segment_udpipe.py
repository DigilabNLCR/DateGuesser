#!/usr/bin/env python
"""This script takes a UDPipe output XML file and adds <passage> tags
to delimit segments of a given target length, measured in tokens. The segmenter,
however, does not split sentences, so the segments will not have the exact
same number of tokens; rather, the segment lengths will converge towards
the given target length on average."""
from __future__ import print_function, unicode_literals
import argparse
import logging
import pprint
import os.path
import time

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


# NOTE: uuid mapper:

uuid_to_year = {'defb0650-affe-11dd-8701-000d606f5dc6': '1874', 'a30280e0-d09f-11dc-a155-000d606f5dc6': '1875', '512c4270-055f-11dd-9584-000d606f5dc6': '1878', '54dc66d0-b4ed-11e9-9209-005056827e51': '1885', '8a2fdf10-f672-11dc-bbc2-000d606f5dc6': '1879', 'bc60e780-be71-11e9-8fdf-005056827e52': '1880', '74192290-bf1a-11e9-8fdf-005056827e52': '1881', '28d2c9e0-b51e-11e9-8fdf-005056827e52': '1881', 'c5602930-e373-11dc-a899-000d606f5dc6': '1882', 'dff52ca0-b50d-11e9-8fdf-005056827e52': '1883', '97ea20c0-8b0e-11de-8062-000d606f5dc6': '1890', '08df98d0-b6ff-11dd-b835-000d606f5dc6': '1883', 'cd18d6b0-12c4-11dd-ab8f-000d606f5dc6': '1883', 'ba63f110-094a-11e5-ae7e-001018b5eb5c': '1885', '7c8272c0-0947-11e5-ae7e-001018b5eb5c': '1889', 'e7551880-1a81-11dd-9082-000d606f5dc6': '1885', 'c7a90bd0-fdb7-11dd-9972-000d606f5dc6': '1885', '59dd5b80-bf45-11dc-8930-000d606f5dc6': '1886', 'b603f190-e8f9-11e4-9c07-001018b5eb5c': '1886', 'cc654660-0d42-11e5-b0b8-5ef3fc9ae867': '1887', '2b411440-a857-11de-b12e-000d606f5dc6': '1887', '99063ab0-f66e-11dc-a1ec-000d606f5dc6': '1888', 'db77b6f0-b6d8-11dd-8fbc-000d606f5dc6': '1888', '24dc0180-01b5-11de-8d54-000d606f5dc6': '1890', '4e4fb6c0-ba06-11dd-8cd4-000d606f5dc6': '1890', '3c5b2730-b7c5-11dd-96e9-000d606f5dc6': '1895', '457424b0-b7c6-11dd-a638-000d606f5dc6': '1898', '11bb9630-d67a-11e7-9c45-005056827e52': '1901', 'afe68830-11c2-11ea-af21-005056827e52': '1907', 'c9de5370-d62f-11dc-acb7-000d606f5dc6': '1896', 'a141f1a0-d630-11dc-895d-000d606f5dc6': '1896', 'cf875600-f2ad-11dd-9680-000d606f5dc6': '1897', '005a1410-9020-11ed-b7c5-5ef3fc9bb22f': '1891', 'b6ec5470-55d5-11de-bc6b-000d606f5dc6': '1891', '271a8310-5668-11de-bb8f-000d606f5dc6': '1892', '6f7f0b10-ef70-11e4-a511-5ef3fc9ae867': '1894', 'd6f02dc0-894b-11dd-ba4a-000d606f5dc6': '1894', '70d31c30-0a5e-11dd-aa2e-000d606f5dc6': '1896', '0c55b3e0-00d7-11e5-93b2-001018b5eb5c': '1897', '527d93b0-d943-11dc-81c2-000d606f5dc6': '1899', '33a057a0-0a5f-11dd-91d7-000d606f5dc6': '1897', 'b4210300-4e73-11eb-b4d1-005056827e51': '1902', '4fd57e30-a2c7-11e3-b833-005056827e52': '1905', 'af1bfa80-283d-11e4-8e0d-005056827e51': '1914', 'a3e9c2d0-5b97-11ef-9a22-5ef3fc9bb22f': '1918', 'fd8b34d0-e6c3-11e8-8d10-5ef3fc9ae867': '1905', 'abead110-7069-11e8-87bd-005056827e52': '1911', '9043fa70-382f-11e4-8e0d-005056827e51': '1913', 'c8283c20-355f-11ef-9fa1-5ef3fc9bb22f': '1913', '2c8396f0-47cc-11e5-a525-5ef3fc9ae867': '1915', '2c518070-0385-11e4-89c6-005056827e51': '1921', 'c00e9800-733d-11e6-81ec-005056827e51': '1921', '015b87b0-0ed8-11e7-968f-005056827e51': '1930', '392d9780-323a-11e6-ae84-005056827e51': '1930', '24a2aec0-7e46-11e5-ac67-005056827e51': '1922', 'dfd67d00-d888-11dc-bb94-000d606f5dc6': '1861', '8ccc7b40-e450-11dc-b48e-000d606f5dc6': '1862', '1baf33b0-14d3-11e5-9192-001018b5eb5c': '1865', 'a0a707f0-cfdf-11dc-af3e-000d606f5dc6': '1866', '9c1005a0-9e45-11dc-abe5-000d606f5dc6': '1869', 'ec38f6b0-320e-11dd-b356-000d606f5dc6': '1872', '8ec9c5d0-0555-11dd-8b32-000d606f5dc6': '1872', 'b1587470-393f-11dd-9696-000d606f5dc6': '1872', 'f837aec0-9938-11dd-b3b4-000d606f5dc6': '1874', '621b80d0-b50e-11e9-8fdf-005056827e52': '1880', 'acb14460-6ae0-11dd-96fe-000d606f5dc6': '1882', 'df270000-7b4f-11eb-9d4f-005056827e52': '1886', 'aa871ec0-094a-11e5-ae7e-001018b5eb5c': '1887', 'c96e6410-f66f-11dc-b23a-000d606f5dc6': '1888', '1aa0e260-ae1d-11ee-a51e-005056827e52': '1889', '01b118f0-0eae-11e5-b269-5ef3fc9bb22f': '1897'}


class XMLConstants:
    SENTENCE_START = '<s'
    SENTENCE_START_TAG = '<s>'
    SENTENCE_END = '</s'
    SENTENCE_END_TAG = '</s>'

    PASSAGE_START = '<passage'
    PASSAGE_END = '</passage'
    PASSAGE_END_TAG = '</passage>'

    PASSAGE_TYPE_TRAIN = 'train'
    PASSAGE_TYPE_DEV = 'devel'
    PASSAGE_TYPE_TEST = 'test'

    DEFAULT_PASSAGE_TYPE = 'train'

    @staticmethod
    def create_passage_start_tag(passage_header, passage_sentences, target_length, passage_type="train"):
        passage_length = sum([len(sentence) for sentence in passage_sentences])
        return '{} id="{}" length="{}" target_length="{}" type="{}">' \
               ''.format(XMLConstants.PASSAGE_START, passage_header, passage_length, target_length, passage_type)

    @staticmethod
    def get_passage_type_from_tag(passage_header_tag):
        for field in passage_header_tag.split(' '):
            if field.startswith('type'):
                return field.split('"')[1]
        # logging.warning('Header tag does not contain type: {}'.format(passage_header_tag))
        return XMLConstants.DEFAULT_PASSAGE_TYPE

def parse_lines_into_passages(lines):
    '''Returns a dictionary of passages. Each passage is indexed by its ID.'''
    passages = {}

    current_passage_lines = []
    current_passage_header = None
    for line in lines:
        # logging.debug('Line: {}'.format(line))
        if line.startswith(XMLConstants.PASSAGE_START):
            # This includes the type of the passage, if applicable.
            current_passage_header = line.strip()
        elif line.startswith(XMLConstants.PASSAGE_END):
            passages[current_passage_header] = current_passage_lines
            current_passage_lines = []
            current_passage_header = None
        else:
            current_passage_lines.append(line.strip())

    return passages


def parse_lines_into_sentences(lines):
    sentences = []

    # logging.debug('Lines: {}'.format(len(lines)))
    # logging.debug('Lines: {}'.format(type(lines)))

    current_sentence = []
    for line in lines:
        # logging.debug('Line: {}'.format(line))
        if line.startswith(XMLConstants.PASSAGE_START) or line.startswith(XMLConstants.PASSAGE_END):
            # Ignore existing passages.
            continue
        if line.startswith(XMLConstants.SENTENCE_START):
            current_sentence = []
        elif line.startswith(XMLConstants.SENTENCE_END):
            sentences.append(current_sentence)
            current_sentence = []
        else:
            current_sentence.append(line.strip())

    return sentences


def build_passage_header(author_code, book_code, year, target_length, passage_number):
    if author_code.startswith('-'):
        raise ValueError('Invalid author code {}: cannot start with hyphen.'.format(author_code))
    if book_code.startswith('-'):
        raise ValueError('Invalid book code {}: cannot start with hyphen.'.format(book_code))
    if year.startswith('-'):
        raise ValueError('Invalid preprocessing code {}: cannot start with hyphen.'.format(year))

    return 'a-{}.b-{}.y-{}.s-{}.p-{}'.format(author_code,
                                             book_code,
                                             year,
                                             target_length,
                                             passage_number)


def segment_sentences_into_passages(sentences, target_length=100, allow_leftover=True,
                                    author_code=None, book_code=None, year=None,
                                    output_passages_type='train',
                                    starting_passage_number=0):
    '''Here the sentences are just a list of sentences.'''
    passages = {}

    passage_number = starting_passage_number
    current_target_length = target_length
    current_passage_sentences = []
    current_passage_length = 0

    # logging.debug('Working with {} total sentences'.format(len(sentences)))

    for sentence in sentences:
        sentence_length = len(sentence)
        current_passage_sentences.append(sentence)
        current_passage_length += sentence_length

        # logging.info('Sentence: {}'.format(sentence))

        if current_passage_length > current_target_length:
            passage_header = build_passage_header(author_code, book_code, year,
                                                  target_length, passage_number)
            passage_header_tag = XMLConstants.create_passage_start_tag(passage_header, current_passage_sentences,
                                                                       target_length, output_passages_type)

            passages[passage_header_tag] = current_passage_sentences
            current_target_length += target_length - current_passage_length

            current_passage_sentences = []
            current_passage_length = 0
            passage_number += 1

    # Dealing with the last passage.
    if not allow_leftover:
        # TODO: Smarter algorithm for distributing leftover sentences among other passages.
        # If it is less than 1/2 target length, attach to previous. If it is greater, leave as a separate segment.
        if current_passage_length == 0:
            # If there is no leftover, nothing to do.
            pass
        elif current_passage_length < (target_length / 2):
            last_passage_number = passage_number
            if passage_number > 0:
                last_passage_number = passage_number - 1

            last_passage_header = build_passage_header(author_code, book_code, year, target_length, last_passage_number)
            last_passage_header_tag = XMLConstants.create_passage_start_tag(last_passage_header, current_passage_sentences,
                                                                       target_length, output_passages_type)

            logging.info('Last passage header tag: {}'.format(last_passage_header_tag))
            logging.info('Passage headers: {}'.format(passages.keys()))

            # The length of the leftover does not match the length of the already stored last
            # passage, so because the whole passage tag is the passages dict key here,
            # we need to sub-select by ID.
            prev_passage_header_tag = [k for k in passages.keys() if k.startswith(last_passage_header_tag.split()[0])][0]
            passages[prev_passage_header_tag].extend(current_passage_sentences)
        else:
            last_passage_header = build_passage_header(author_code, book_code, year, target_length, passage_number)
            last_passage_header_tag = XMLConstants.create_passage_start_tag(last_passage_header, current_passage_sentences,
                                                                       target_length, output_passages_type)
            passages[last_passage_header_tag] = current_passage_sentences
    else:
        # If leftovers are allowed, discard the leftover.
        pass

    return passages


def segment_sentences_dict_into_passages(sentences_dict, target_length=100, allow_leftover=True,
                                         author_code=None, book_code=None, year=None,
                                         starting_passage_number=0):

    print('sentences_dict :', len(sentences_dict['ALL']))
    
    passages = {}
    current_starting_passage_number = starting_passage_number
    for i, (input_passage_header, sentences) in enumerate(sentences_dict.items()):
        # The passage type is in the header here, but not in the output header (yet).
        #if i == 0:
        #    logging.info('segment_sentences_dict_into_passages: passage header {}'.format(input_passage_header))
        passage_type = XMLConstants.get_passage_type_from_tag(input_passage_header)
        print('Passage {}: passage type {}'.format(input_passage_header, passage_type))
        current_passages = segment_sentences_into_passages(sentences,
                                                           target_length=target_length,
                                                           allow_leftover=allow_leftover,
                                                           author_code=author_code,
                                                           book_code=book_code,
                                                           year=year,
                                                           output_passages_type=passage_type,
                                                           starting_passage_number=current_starting_passage_number)
        current_starting_passage_number += len(current_passages)
        for current_passage_header in current_passages:
            passages[current_passage_header] = current_passages[current_passage_header]

    print('passages :', len(passages))
    
    return passages


def print_passages_to_file(filename, passages, target_length):
    '''The passages data structure is a dictionary. Keys are passage IDs, values are lists of sentences.
    Sentences are lists of tokens.

    Output is rendered as XML:

    <passage id="..." target_length="..." length="...">
        <s>
            <w n="1" ...>Russian</w>
            <w n="2" ...>warship</w>
            <pc n="3">,</ps>
            <w n="4" ...>go</w>
        </s>
    </passage>

    '''
    output_file_lines = []
    for passage_header_tag, passage_sentences in passages.items():

        # This should be moved into the sentences dict creation...
        # passage_tag = XMLConstants.create_passage_start_tag(passage_header,
        #                                                     passage_sentences,
        #                                                     target_length=target_length)
        output_file_lines.append(passage_header_tag)

        for sentence in passage_sentences:
            output_file_lines.append('\t' + XMLConstants.SENTENCE_START_TAG)
            for token in sentence:
                output_file_lines.append('\t\t' + token)
            output_file_lines.append('\t' + XMLConstants.SENTENCE_END_TAG)

        output_file_lines.append(XMLConstants.PASSAGE_END_TAG)

    with open(filename, 'w', encoding='utf-8') as fh:
        fh.writelines([l + '\n' for l in output_file_lines])



def verify_passage_resegmentation_boundaries(passages_source, passages_resegmented):
    """Checks that the source passages have been resegmented in a manner that respects
    the original passages, leaves out no tokens, and has the same sentence boundaries.

    Both `passages_source` and `passages_segmented` are what comes out of:

    ```
        sentences_dict = {passage_header: parse_lines_into_sentences(passage_lines)
                     for passage_header, passage_lines in parse_lines_into_passages(lines).items() }
    ```

    called over `lines` as output from `fh.readlines()` on the input XML files. That is,
    the input arguments are dicts of sentences, and each sentence is a list of tokens.
    """

    # Is the number of tokens the same?
    n_tokens_source = 0
    n_tokens_resegmented = 0
    for i, passage_sentences in enumerate(passages_source.values()):
        # logging.debug('Source passage {} has {} sentences'.format(i, len(passage_sentences)))
        if i == (len(passages_source) - 1):
            logging.debug('Last source sentence:\n{}'.format(pprint.pformat(passage_sentences[-1])))

        for s in passage_sentences:
            n_tokens_source += len(s)

    for i, passage_sentences in enumerate(passages_resegmented.values()):
        # logging.debug('Source passage {} has {} sentences'.format(i, len(passage_sentences)))
        if i == (len(passages_resegmented) - 1):
            logging.debug('Last resegmented sentence:\n{}'.format(pprint.pformat(passage_sentences[-1])))

        for s in passage_sentences:
            n_tokens_resegmented += len(s)

    if n_tokens_source != n_tokens_resegmented:
        logging.info('Number of source ({}) and resegmented ({}) tokens doesn\'t match!'
                     ''.format(n_tokens_source, n_tokens_resegmented))
        return False
    else:
        logging.info('Number of source and resegmented tokens matches: {}'.format(n_tokens_resegmented))

    # Are all last tokens of source passages also last tokens of resegmented passages?
    # This is not straightforward, since the resegmented passages keep no reference
    # to their source passage -- we don't know which last token is which.
    # However, passage IDs are monotonous (numerically), which can be used.
    def _get_passage_number_from_id(passage_id):
        return int(passage_id.split('-')[-1])

    def _get_id_from_passage_tag(passage_line):
        return passage_line.split('"')[1]

    def _get_passage_number_from_tag(passage_line):
        try:
            return _get_passage_number_from_id(_get_id_from_passage_tag(passage_line))
        except IndexError:
            return _get_passage_number_from_id(passage_line)

    sorted_source_passage_last_lines = [passages_source[passage][-1][-1]
                                        for passage in sorted(passages_source,
                                                              key=lambda k: _get_passage_number_from_tag(k))]
    sorted_resegmented_passage_last_lines = [passages_resegmented[passage][-1][-1]
                                             for passage in sorted(passages_resegmented,
                                                                   key=lambda k: _get_passage_number_from_tag(k))]

    source_passage_idx = 0
    for resegmented_passage_idx, passage_last_line in enumerate(sorted_resegmented_passage_last_lines):
        # if resegmented_passage_idx == 1:
            # logging.info('Reseg. passage last line: {}'.format(sorted_resegmented_passage_last_lines))
            # logging.info('Source passage last line: {}'.format(sorted_source_passage_last_lines[source_passage_idx]))
        if passage_last_line.strip() == sorted_source_passage_last_lines[source_passage_idx].strip():
            source_passage_idx += 1

    if source_passage_idx < len(passages_source):
        logging.info('Not all last lines of source passages found!'
                     ' Resegmentation is suspicious. Missing: {}'.format(len(passages_source) - source_passage_idx))
        return False

    return True


def _infer_auto_header(input_xml_filename: str):
    '''Return the auuthor_code, book_code and preprocessing_code
    triplet. To be used when re-segmenting, to save having to infer these
    externally.'''
    filename_nopath = os.path.basename(input_xml_filename)
    fields = filename_nopath.split('.')
    author_field, book_field, preprocessing_field = fields[1], fields[2], fields[3]
    author_code = author_field.split('-')[-1]
    book_code = book_field.split('-')[-1]
    preprocessing_code = preprocessing_field.split('-')[-1]
    return author_code, book_code, preprocessing_code


def _build_auto_output_xml_file(input_xml_filename: str, output_dir: str, target_length: int):
    a, b, r = _infer_auto_header(input_xml_filename)
    input_basename = os.path.basename(input_xml_filename)
    uuid = input_basename.split('.')[0]
    output_basename = '.'.join([uuid,
                                'a-{}'.format(a),
                                'b-{}'.format(b),
                                'r-{}'.format(r),
                                's-{}'.format(target_length),
                                'xml'])
    output_filename = os.path.join(output_dir, output_basename)
    return output_filename

#####################################################################################


# def build_argument_parser():
#     parser = argparse.ArgumentParser(description=__doc__, add_help=True,
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)

#     parser.add_argument('-i', '--input_xml', action='store', required=True,
#                         help='XML file that should be segmented. Expects udpipe output format.')
#     parser.add_argument('-o', '--output_xml', action='store', required=False,
#                         help='Name of XML file to which the output will be written.')

#     parser.add_argument('-a', '--author_code', action='store',
#                         help='Author code used to generate passage IDs.')
#     parser.add_argument('-b', '--book_code', action='store',
#                         help='Book code used to generate passage IDs.')
#     parser.add_argument('-r', '--preprocessing_code', action='store',
#                         help='Preprocessing code to generate passage IDs.')
#     parser.add_argument('-t', '--target_length', action='store', type=int,
#                         help='Target average segment length.')

#     parser.add_argument('--use_existing_segmentation', action='store_true',
#                         help='If set, will respect existing <passage> boundaries'
#                              ' and will not discard anything at the end.')
#     parser.add_argument('--auto_header', action='store_true',
#                         help='If running re-segmentation from a file which was'
#                              ' created with an author/book/preprocessing/segment length'
#                              ' header by this script, use this to auto-infer these'
#                              ' values from the header (except for target length)'
#                              ' when re-segmenting with --use_existing_segmentation.')

#     parser.add_argument('--auto_output_name', action='store_true',
#                         help='If set when resegmenting, will 1) derive the output XML filename from the input name'
#                              ' plus target length, 2) use the --auto_output_dir argument to determine'
#                              ' which directory the output should be written with the auto-derived filename.'
#                              ' Use only with --use_existing_segmentation.')
#     parser.add_argument('--auto_output_dir', action='store',
#                         help='In case the output XML name is derived automatically, the output file'
#                              ' is placed into this directory. The directory must already exist.'
#                              ' If --auto_output_name is used, this argument must be set')

#     parser.add_argument('-v', '--verbose', action='store_true',
#                         help='Turn on INFO messages.')
#     parser.add_argument('--debug', action='store_true',
#                         help='Turn on DEBUG messages.')

#     return parser


# NOTE: change args to input_xml, output_xml, target_length, author_code, book_code, preprocessing_code
def main(input_xml, output_xml, target_length, author_code, book_uuid, year, use_existing_segmentation=False):
    logging.info('Starting main...')
    _start_time = time.process_time()

    # Your code goes here
    with open(input_xml, encoding='utf-8') as fh:
        lines = fh.readlines()

    # passages = {}
    if use_existing_segmentation:
        input_passages = parse_lines_into_passages(lines)
        logging.debug('Total input passages after parse_lines_into_passages: {}'.format(len(input_passages)))
    else:
        input_passages = {'ALL': parse_lines_into_sentences(lines)}
        # print('num_of_input_sentences_PASS :', len(input_passages['ALL']))

    # sentences = {}
    if (not use_existing_segmentation) or (len(input_passages) == 0):
        # When existing segmentation is not supposed to be conserved, we just
        # re-segment everything, so we use directly parse_lines_into_sentences()
        # (which discards passage information).
        sentences_dict = {'ALL': parse_lines_into_sentences(lines)}
        # print('num_of_input_sentences_SENT :', len(sentences_dict['ALL']))
    else:
        sentences_dict = { passage_header: parse_lines_into_sentences(passage_lines)
                           for passage_header, passage_lines in input_passages.items() }


    # NOTE: header auto-inference is not implemented here
    # if auto_header and use_existing_segmentation:
    #     author_code, book_code, preprocessing_code = _infer_auto_header(input_xml)
    #     logging.info('Auto-header inference: {}, {}, {}'
    #                  ''.format(author_code, book_code, preprocessing_code))

    allow_leftover = not use_existing_segmentation
    # Here we need to keep the passage types as well. This is, fortunately, kept as part of the
    # sentences_dict keys, since they use the string of the whole passage tag (with all attributes).
    # This was verified with:
    # logging.debug('Example sentences_dict key:\n{}'.format(list(sentences_dict.keys())[0]))
    passages = segment_sentences_dict_into_passages(sentences_dict,
                                               target_length=target_length,
                                               allow_leftover=allow_leftover,
                                               author_code=author_code,
                                               book_code=book_uuid,
                                               year=year)

    # Check
    if use_existing_segmentation:
        # logging.debug('Input passages: {}, resegmented passages: {}'.format(len(sentences_dict), len(passages)))
        try:
            if not verify_passage_resegmentation_boundaries(sentences_dict, passages):
                logging.warning('Passage resegmentation is suspicious! Perform manual check of result.')
        except Exception as e:
            logging.warning('Exception occured during resegmentation validity check: {} Check result manually.'.format(e))

    # Output name auto-generation?
    # NOTE: not implemented here
    # if use_existing_segmentation and auto_output_name:
    #     if not auto_output_dir:
    #         raise ValueError('The --auto_output_dir argument must be set when using --auto_output_name!')
    #     output_xml = _build_auto_output_xml_file(input_xml, auto_output_dir, target_length)
    #     logging.info('Auto-build output XML: {}'.format(output_xml))

    print_passages_to_file(output_xml, passages, target_length=target_length)

    _end_time = time.process_time()
    logging.info('segment_udpipe.py done in {0:.3f} s'.format(_end_time - _start_time))

# def main(args):
#     logging.info('Starting main...')
#     _start_time = time.process_time()

#     # Your code goes here
#     with open(input_xml, encoding='utf-8') as fh:
#         lines = fh.readlines()

#     # passages = {}
#     if use_existing_segmentation:
#         input_passages = parse_lines_into_passages(lines)
#         logging.debug('Total input passages after parse_lines_into_passages: {}'.format(len(input_passages)))
#     else:
#         input_passages = {'ALL': parse_lines_into_sentences(lines)}

#     # sentences = {}
#     if (not use_existing_segmentation) or (len(input_passages) == 0):
#         # When existing segmentation is not supposed to be conserved, we just
#         # re-segment everything, so we use directly parse_lines_into_sentences()
#         # (which discards passage information).
#         sentences_dict = {'ALL': parse_lines_into_sentences(lines)}
#     else:
#         sentences_dict = { passage_header: parse_lines_into_sentences(passage_lines)
#                            for passage_header, passage_lines in input_passages.items() }


#     if auto_header and use_existing_segmentation:
#         author_code, book_code, preprocessing_code = _infer_auto_header(input_xml)
#         logging.info('Auto-header inference: {}, {}, {}'
#                      ''.format(author_code, book_code, preprocessing_code))

#     allow_leftover = not use_existing_segmentation
#     # Here we need to keep the passage types as well. This is, fortunately, kept as part of the
#     # sentences_dict keys, since they use the string of the whole passage tag (with all attributes).
#     # This was verified with:
#     # logging.debug('Example sentences_dict key:\n{}'.format(list(sentences_dict.keys())[0]))
#     passages = segment_sentences_dict_into_passages(sentences_dict,
#                                                target_length=target_length,
#                                                allow_leftover=allow_leftover,
#                                                author_code=author_code,
#                                                book_code=book_code,
#                                                preprocessing_code=preprocessing_code)

#     # Check
#     if use_existing_segmentation:
#         # logging.debug('Input passages: {}, resegmented passages: {}'.format(len(sentences_dict), len(passages)))
#         try:
#             if not verify_passage_resegmentation_boundaries(sentences_dict, passages):
#                 logging.warning('Passage resegmentation is suspicious! Perform manual check of result.')
#         except Exception as e:
#             logging.warning('Exception occured during resegmentation validity check: {} Check result manually.'.format(e))

#     # Output name auto-generation?
#     if use_existing_segmentation and auto_output_name:
#         if not auto_output_dir:
#             raise ValueError('The --auto_output_dir argument must be set when using --auto_output_name!')
#         output_xml = _build_auto_output_xml_file(input_xml, auto_output_dir, target_length)
#         logging.info('Auto-build output XML: {}'.format(output_xml))

#     print_passages_to_file(output_xml, passages, target_length=target_length)

#     _end_time = time.process_time()
#     logging.info('segment_udpipe.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    # parser = build_argument_parser()
    # args = parser.parse_args()

    # if verbose:
    #     logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    # if debug:
    #     logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    # main(args)

    # NOTE: random splitting multiple files.

    INPUT_FOLDER = 'Dataset/Svetla_final/'
    OUTPUT_FOLDER = 'Dataset/Svetla_passages/'

    for xml_file in os.listdir(INPUT_FOLDER):
        input_xml = os.path.join(INPUT_FOLDER, xml_file)
        output_xml = os.path.join(OUTPUT_FOLDER, xml_file)

        uuid = xml_file.split('.')[0]

        with open(input_xml, encoding='utf-8') as fh:
            data = fh.read()
            sentence_count = data.count('<s>')
            sentence_end_count = data.count('</s>')
            print('Sentence count:', sentence_count, sentence_end_count, uuid)

        main(input_xml, output_xml, target_length=500, author_code='svetla', book_uuid=uuid, year=uuid_to_year[uuid], use_existing_segmentation=False)
