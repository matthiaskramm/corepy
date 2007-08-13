import inspect

print "You're at:", inspect.stack()[0][2]
print "You're at:", inspect.stack()[0][2]
print "You're at:", inspect.stack()[0][2]
print "You're at:", inspect.stack()[0][2]
print "You're at:", inspect.stack()[0][2]

def foo():
    print "Foo at:", inspect.stack()[0][2]

foo()
foo()
