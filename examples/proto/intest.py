
class Foo(object):
  def __init__(self, *ops):
    super(Foo, self).__init__(*ops)
    print 'Foo'


class Bar(object):
  def __init__(self, b, *ops):
    super(Bar, self).__init__(*((b,)+ops))
    print 'Bar', b


class Baz(Foo, Bar): pass
#   def __init__(self):
#     super(Baz, self).__init__()
#    print 'Baz'


b = Baz(1)
# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

