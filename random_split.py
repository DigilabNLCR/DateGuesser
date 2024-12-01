"""This is a script that assigns types (train/dev/test) to passages randomly
according to the given proportion (by default: 20 % test, 20 % dev, 60 % train).
This should only be run once per dataset, so that the dataset stays consistent.
To this end, if there are <passage> tags detected that already have a type
in the input file, no output is produced.
"""
from __future__ import print_function, unicode_literals
import argparse
import logging
import time

import random


__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def set_passage_types(lines, proportion_test, proportion_dev):
    """Assigns to each passage its output type. Wrapper around everything
    done in this script. Returns a dict of passage header tag --> type pairs."""
    passage_tags = []
    for line in lines:
        if line.startswith('<passage'):
            passage_tags.append(line)

    random.shuffle(passage_tags)

    n_test = int(len(passage_tags) * proportion_test)
    n_dev = int(len(passage_tags) * proportion_dev)

    test_tags = passage_tags[:n_test]
    dev_tags = passage_tags[n_test:n_test + n_dev]
    train_tags = passage_tags[n_test + n_dev:]

    passage_types_dict = {}
    for t in test_tags:
        passage_types_dict[t] = 'test'
    for d in dev_tags:
        passage_types_dict[d] = 'devel'
    for t in train_tags:
        passage_types_dict[t] = 'train'

    return passage_types_dict



def apply_passage_types(lines, types_per_passage_header):
    output_lines = []
    for line in lines:
        if line.startswith('<passage'):
            passage_type = types_per_passage_header[line]
            # NOTE: František: I have made edit here, the original duplicated the 'type="..."' part of <passage ...> tag ... DUNNO why, before it has worked well...
            output_line = line.replace(f'type="train"', f'type="{passage_type}"')
            # output_line = line[:-1] + ' ' + 'type="{}"'.format(passage_type) + '>'
            output_lines.append(output_line)
        else:
            output_lines.append(line)
    return output_lines


################################################################################

def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input_xml', action='store', required=True,
                        help='XML file that should be segmented. Expects udpipe output format.')
    parser.add_argument('-o', '--output_xml', action='store', required=False,
                        help='Name of XML file to which the output will be written.')

    parser.add_argument('-t', '--proportion_test', type=float, action='store', default=0.2)
    parser.add_argument('-d', '--proportion_dev', type=float, action='store', default=0.2)

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.process_time()

    if args.proportion_test > 1:
        raise ValueError('Test proportion must be between 0 and 1! Is: {}'.format(args.proportion_test))
    if args.proportion_dev > 1:
        raise ValueError('Dev proportion must be between 0 and 1! Is: {}'.format(args.proportion_dev))

    # Your code goes here
    with open(args.input_xml, encoding='utf-8') as fh:
        lines = fh.readlines()
        lines = [l.rstrip() for l in lines]   # strip newline

    types_per_passage_header = set_passage_types(lines,
                                                 args.proportion_test,
                                                 args.proportion_dev)

    lines_with_passage_types = apply_passage_types(lines, types_per_passage_header)

    with open(args.output_xml, 'w', encoding='utf-8') as fh:
        fh.writelines([l + '\n' for l in lines_with_passage_types])


    _end_time = time.process_time()
    logging.info('random_split_udpipe.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
