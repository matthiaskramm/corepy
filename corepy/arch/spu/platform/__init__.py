
import corepy.corepy_conf as conf

platform_imports = [
  'Processor', 'InstructionStream', 'ParallelInstructionStream', 'aligned_memory',
  'WORD_SIZE', 'WORD_TYPE', 'spu_exec',
  'SPURegister']


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
  
class _Empty: pass

synbuffer = _Empty()
