import nextarray
#import numpy

ne = nextarray.nextarray('i')
nex = nextarray.nextarray('L', 16)
nex[3] = 1
nex[0] = 2
nex[7] = 3
print "len", len(ne), nex[3], nex[0], nex[7]

