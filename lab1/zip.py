#!/usr/bin/env python
import sys

with open(sys.argv[1]) as fil:
        for line, i in zip(fil, [2 ** x for x in xrange(0, 18)]):
                print i, " ", line
