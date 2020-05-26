#!/bin/python3 -i

import collections
import csv
import hazm
import pandas
import sys

from typing import Counter, Dict, List, Tuple

def collect_adjective_chains(sentences : List[str]):
    adj_chains = {}
    tagger = hazm.POSTagger(model='resources/postagger.model')

    for s in sentences:
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

def export_adj_chains(outfile, adj_chains, chain_length : int):
    writer = csv.writer(outfile)
    writer.writerow(['Adj ' + str(i+1) for i in range(chain_length)] + ['Frequency'])
    writer.writerows([[word for word in chain] + [count] for chain, count in adj_chains[chain_length].items()])

def analyze_adjective_position(adj_chains : Dict[int, Counter[str]]):
    data = collections.Counter()

    for chain_length in adj_chains.keys():
        for chain, chain_freq in adj_chains[chain_length].items():
            for position, word in enumerate(chain):
                data[(chain_length, position + 1, word)] += chain_freq

    df = pandas.DataFrame(columns=['chain-length', 'position', 'frequency', 'word'], data=([cl, p, f, w] for (cl, p, w), f in data.items()))
    return df, data

#TODO: Allow for multiple files
if len(sys.argv) != 2:
    print("Expected a filename")
    sys.exit(1)

corpus_text = ""

with open(sys.argv[1]) as infile:
        corpus_text = '\n'.join(infile.readlines())

sentences = hazm.sent_tokenize(corpus_text)
adj_chains = collect_adjective_chains(sentences)

