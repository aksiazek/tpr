import codecs

M = 50
N = 100
P = 50

with codecs.open('file2.txt', 'w', 'utf-8') as _file:
        num = 1.0
        for i in xrange(M):
                for j in xrange(N):
                        if i == 0 and j == 0:
                                continue
                        _file.write("A,"+str(i)+','+str(j)+','+str(num)+'\n')
                        num += 1.0
        num = 1.0
        for i in xrange(N):
                for j in xrange(P):
                        if i == 0 and j == 0:
                                continue
                        _file.write("B,"+str(i)+','+str(j)+','+str(num)+'\n')
                        num += 1.0
