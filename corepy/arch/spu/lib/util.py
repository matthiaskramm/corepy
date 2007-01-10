
import corepy.arch.spu.isa as spu

def load_word(code, r_target, word, clear = False, zero = True):
  # If r0 != 0, set zero to false
  if zero and (-512 < word < 511):
    code.add(spu.ai(word, 0, r_target))
  else:
    # This method will work for integers, but...
    # code.add(spu.ilhu(word / pow(2, 16), r_target))
    code.add(spu.ilhu((word & 0xFFFF0000) >> 16, r_target))
    # code.add(spu.ilhu(word % pow(2, 16), r_target))
    code.add(spu.iohl((word & 0xFFFF), r_target))

  if clear:
    code.add(spu.shlqbyi(12, r_target, r_target))
  return




