#!/bin/python3 -i
"""Analyze Persian adjective use

Usage:
  analyze.py -r <file>...
  analyze.py -l <file>
  analyze.py -h

Options:
  -h --help                       Show this screen.
  -r --read                       Read in files.
  -l --load                       Load in a dump.

"""

import collections
import csv
import hazm
import math
import pandas
import pickle 
import random
import sys

from docopt import docopt
from typing import Counter, Dict, List, Tuple

def collect_adjective_chains(sentences : List[str], progress = False):
    adj_chains = {}
    tagger = hazm.POSTagger(model='resources/postagger.model')

    numSentences = len(sentences)

    for i,s in enumerate(sentences):
        if progress and i % (numSentences//200) == 0:
            print(f'Sentences analyzed: {100.0*i/numSentences}%, {i}/{numSentences}')
        tags = tagger.tag(hazm.word_tokenize(s))
        chain = []
        prevWordAdj = False
        for word, tag in tags:
            if prevWordAdj:
                if tag == 'AJ':
                    if word == 'تر':
                        chain[-1] += word
                    else:
                        chain.append(word)
                else:
                    # if the word isn't an adjective, the chain is done
                    # append the chain to the list of n-chains

                    length = len(chain)
                    if adj_chains.get(length) == None:
                        adj_chains[length] = collections.Counter([tuple(chain)])
                    else:
                        adj_chains[length][tuple(chain)] += 1

                    prevWordAdj = False
            elif tag == 'AJ':
                chain = [word]
                prevWordAdj = True

        # if the last word in the sentence was an adjective, we have one more chain to add
        if prevWordAdj:
                length = len(chain)
                if adj_chains.get(length) == None:
                    adj_chains[length] = collections.Counter([tuple(chain)])
                else:
                    adj_chains[length][tuple(chain)] += 1

    return adj_chains

def create_adj_position_df(adj_chains : Dict[int, Counter[str]]):
    data = collections.Counter()

    for chain_length in adj_chains.keys():
        for chain, chain_freq in adj_chains[chain_length].items():
            for position, word in enumerate(chain):
                data[(chain_length, position + 1, word)] += chain_freq

    df = pandas.DataFrame(columns=['chain-length', 'position', 'frequency', 'word'], data=([cl, p, f, w] for (cl, p, w), f in data.items()))
    return df, data

def create_markov_chain_df(adj_chains : Dict[int, Counter[str]], normalise=True):
    # Map each word to a number, starting with the null marker as 0
    mapping = {None: 0}

    # Build the map while counting adjectives (plus the null marker)
    adj_count = 1
    for chain_length in adj_chains.keys():
        if chain_length > 1:
            for chain, _ in adj_chains[chain_length].items():
                for word in chain:
                    # If the word has not yet been encountered, add it to the next empty index in the mapping
                    if mapping.get(word) is None:
                        mapping[word] = adj_count
                        adj_count += 1

    # One row and column for each of the adjectives (plus the null marker)
    # Each row will contain the probability that each adjective follows that # adjective (or the null marker, if the adjective began the chain)
    data = [[0.0 for _ in range(adj_count)] for _ in range(adj_count)]

    for chain_length in adj_chains.keys():
        if chain_length > 1:
            for chain, chain_freq in adj_chains[chain_length].items():
                prev_word_index = 0
                for word in chain:
                    word_index = mapping[word]
                    data[prev_word_index][word_index] += chain_freq
                    prev_word_index = word_index

    def normalise_func(row : pandas.core.series.Series):
        total = row.sum()
        return row / total if total > 0 else 0.0

    adjs = list(mapping.keys())
    df = pandas.DataFrame(columns=adjs, index=adjs, data=data)
    if normalise:
        df = df.apply(normalise_func, axis=1)

    return df

def cost_function(markov_chain_list, adj_group_vector : List[int]):
    cost = 0
    num_adj = len(adj_group_vector)

    if num_adj != len(markov_chain_df.index):
        raise Exception("Adjective grouping vector length does not match the number of adjectives")

    for first_word_index in range(num_adj):
        for second_word_index in range(num_adj):
            # If the preceeding word's group is lower in precedence than
            # the following word's then add to the cost
            if adj_group_vector[first_word_index] > adj_group_vector[second_word_index]:
                cost += markov_chain_list[first_word_index][second_word_index]
            elif adj_group_vector[first_word_index] == adj_group_vector[second_word_index]:
                cost += 0.25 * markov_chain_list[first_word_index][second_word_index]

    num_groups = max(adj_group_vector) + (1 if min(adj_group_vector) == 0 else 0)
    if num_groups > 2:
        cost *= math.log(num_groups)

    return cost

def export_adj_chains(filename, adj_chains, chain_length : int):
    with open(filename, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Adj ' + str(i+1) for i in range(chain_length)] + ['Frequency'])
        writer.writerows([[word for word in chain] + [count] for chain, count in adj_chains[chain_length].items()])

def export_adj_position_df(filename, df):
    with open(filename, 'w') as outfile:
        df_ = df[df['chain-length'] > 1]
        outfile.write(df_.to_csv(index=False))

def serialize_adj_chains(adj_chains, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(adj_chains, outfile, protocol=pickle.HIGHEST_PROTOCOL)

# Main code
arguments = docopt(__doc__, version='Persian Adjective Analysis 0.1')
adj_chains = None
markov_chain_df = None

if arguments['--read']:
    filenames = arguments['<file>']
    corpus_text = ""

    for i, name in enumerate(filenames):
        with open(name) as infile:
            corpus_text += '\n'.join(infile.readlines())
            print(f'Finished reading file {i+1}/{len(filenames)}:\t{name}')

    print('Tokenizing sentences: ', end='')
    sentences = hazm.sent_tokenize(corpus_text)
    print(f'{len(sentences)} total')

    print('Analyzing adjective usage...')
    adj_chains = collect_adjective_chains(sentences, progress=True)
elif arguments['--load']:
    filename = arguments['<file>'][0]
    with open(filename, 'rb') as infile:
        adj_chains = pickle.load(infile)

markov_chain_df = create_markov_chain_df(adj_chains, normalise=False)
markov_chain_list = markov_chain_df.values.tolist()

