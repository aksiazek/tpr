#!/usr/bin/env python

from __future__ import print_function

with open("group-3.dat") as f1:
        with open("group-5.dat") as f2:
                with open("group-10.dat") as f3:
                        for i, a, b, c in zip(['own-bcast','bcast','own-reduce','reduce'], f1,f2,f3):
                                with open("results-"+str(i)+".dat","a+") as out:
                                        out.write('{0} {1}'.format(3, a))
                                        out.write('{0} {1}'.format(5, b))
                                        out.write('{0} {1}'.format(10, c))
