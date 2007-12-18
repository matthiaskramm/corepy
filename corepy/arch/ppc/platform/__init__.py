# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

import corepy.corepy_conf as conf

platform_imports = [
  'Processor', 'InstructionStream',
  'WORD_SIZE', 'WORD_TYPE', 'ExecParams',
  'GPRegister', 'FPRegister', 'VMXRegister']

platform_string = '%(os)s.spre_%(os)s_%(arch)s_%(bits)d' % {
  'os': conf.OS, 'arch': conf.ARCH, 'bits': conf.BITS}

if conf.OS == 'osx':
  platform_imports.append('array_address')
  
print 'Platform:', platform_string
platform_module = __import__(platform_string, globals(), locals(), platform_imports)

for cls in platform_imports:
  try:
    locals()[cls] = getattr(platform_module, cls)
  except:
    print 'PPC Platform Warning: Unable to load osx.ppc_exec.%s.  Related features will not be available.' % (str(cls),)
  
