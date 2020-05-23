#!/bin/python3 -i

import hazm
import sys

if len(sys.argv) != 2:
    print("Expected a filename")
    sys.exit(1)

corpus_text = ""

with open(sys.argv[1]) as infile:
        corpus_text = '\n'.join(infile.readlines())

