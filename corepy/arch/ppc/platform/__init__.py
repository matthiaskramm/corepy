
import corepy.corepy_conf as conf

platform_imports = [
  'Processor', 'InstructionStream',
  'WORD_SIZE', 'WORD_TYPE',
  'GPRegister', 'FPRegister', 'VMXRegister']

platform_string = '%(os)s.spre_%(os)s_%(arch)s_%(bits)d' % {
  'os': conf.OS, 'arch': conf.ARCH, 'bits': conf.BITS}

platform_module = __import__(platform_string, globals(), locals(), platform_imports)

for cls in platform_imports:
  locals()[cls] = getattr(platform_module, cls)
  
