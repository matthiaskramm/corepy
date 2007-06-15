# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)


import corepy.corepy_conf as conf

platform_imports = [
  'Processor', 'InstructionStream', 'ParallelInstructionStream', 'aligned_memory',
  'WORD_SIZE', 'WORD_TYPE', 'spu_exec', 'synbuffer', 'NativeInstructionStream',
  'SPURegister', 'cell_fb']


if conf.OS == 'linux':
  platform_string = 'linux.spre_linux_spu'
elif conf.OS == 'linux_spufs':
  platform_string = 'linux_spufs.spre_linux_spu'
else:
  platform_string = 'spre_dummy_spu'


print 'Platform:', platform_string
platform_module = __import__(platform_string, globals(), locals(), platform_imports)

for cls in platform_imports:
  locals()[cls] = getattr(platform_module, cls)
  
# class _Empty: pass
# synbuffer = _Empty()

