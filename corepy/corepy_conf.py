
import os

sys_info = os.uname()

# Very basic archticture detection...
if sys_info[0] == 'Darwin' and sys_info[-1] == 'Power Macintosh':
  OS = 'osx'
  ARCH = 'ppc'
  BITS = 32
elif sys_info[0] == 'Linux' and sys_info[-1] == 'ppc64':
  OS = 'linux'
  ARCH = 'ppc'
  BITS = 64

  cpus = [line.split(':')[1] for line in open('/proc/cpuinfo').readlines() if line[:3] == 'cpu']
  if len(cpus) > 0 and cpus[0][:5] == ' Cell':
    ARCH = 'cell'
    # OS = 'linux_spufs'
else:
  print "Unsupported architecture: Using 'dummy' settings"
  OS = 'dummy'
  ARCH = 'dummy'
  BITS = 0
