# ------------------------------
# Helpers
# ------------------------------

# Dec->Binary format converter from:
#  http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/219300
bstr_pos = lambda n: n>0 and bstr_pos(n>>1)+str(n&1) or ''
def DecToBin(n):
  """
  Convert an integer into binary coded string.
  """
  s = bstr_pos(n)
  if len(s) < 32:
    s = '0' * (32 - len(s)) + s
  return s

def BinToDec(b):
  """
  Convert a binary coded string to a decimal integer
  """

  l = list(b)
  l.reverse()
  p = 1
  d = 0
  for bit in l:
    d += p * int(bit)
    p = p << 1
  return d

