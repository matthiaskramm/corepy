
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
