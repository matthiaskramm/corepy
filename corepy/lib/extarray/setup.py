from distutils.core import setup, Extension

module1 = Extension('nextarray',
                    sources = ['nextarray.c'])

setup (name = 'nextarray',
       version = '1.0',
       description = 'nextarray',
       ext_modules = [module1])

