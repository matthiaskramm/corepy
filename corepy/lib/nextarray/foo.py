import nextarray
#import numpy

ne = nextarray.nextarray('I', range(0, 8))
print ne
ne[3] = 17
ne[2] = 16
for i in ne:
  print "ne iter", i

nex = nextarray.nextarray('L', 16)
nex[3] = 1
nex[0] = 2
nex[7] = 3
print "len", len(ne), nex[3], nex[0], nex[7]

