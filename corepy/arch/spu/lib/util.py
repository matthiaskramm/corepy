
import corepy.arch.spu.isa as spu

def load_word(code, r_target, word, clear = False, zero = True):
  # If r0 != 0, set zero to false
  if zero and (-512 < word < 511):
    code.add(spu.ai(r_target, 0, word))
  else:
    # This method will work for integers, but...
    # code.add(spu.ilhu(word / pow(2, 16), r_target))
    code.add(spu.ilhu(r_target, (word & 0xFFFF0000) >> 16))
    # code.add(spu.ilhu(word % pow(2, 16), r_target))
    code.add(spu.iohl(r_target, (word & 0xFFFF)))

  if clear:
    code.add(spu.shlqbyi(r_target, r_target, 12))
  return




