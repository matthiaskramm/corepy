# Copyright (c) 2006-2009 The Trustees of Indiana University.                   
# All rights reserved.                                                          
#                                                                               
# Redistribution and use in source and binary forms, with or without            
# modification, are permitted provided that the following conditions are met:   
#                                                                               
# - Redistributions of source code must retain the above copyright notice, this 
#   list of conditions and the following disclaimer.                            
#                                                                               
# - Redistributions in binary form must reproduce the above copyright notice,   
#   this list of conditions and the following disclaimer in the documentation   
#   and/or other materials provided with the distribution.                      
#                                                                               
# - Neither the Indiana University nor the names of its contributors may be used
#   to endorse or promote products derived from this software without specific  
#   prior written permission.                                                   
#                                                                               
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"   
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE     
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE   
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL    
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR    
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER    
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          

from corepy.spre.spe import InstructionOperand, Label, Variable
import corepy.spre.spe as spe
import corepy.arch.ptx.types.registers as regs

# ------------------------------
# x86 Operands
# ------------------------------

class ptxField(InstructionOperand):
  def __init__(self, name, default = None):
    InstructionOperand.__init__(self, name, default)

  def __eq__(self, other):
    return type(self) == type(other)

  def check(self, value):
    return isinstance(value, self) and self.name == value.name

  def render(self):
    return name

class ptxFlagField(ptxField):
  def __init__(self, name, ptxstr, default = None):
    self.ptxstr = ptxstr
    ptxField.__init__(self, name, default)

  def check(self, value):
    return value in (True, False)

  def render(self, value):
    if value:
      return self.ptxstr
    return ''

  def __eq__(self, other):
    return type(self) == type(other) and self.name == other.name

class ptxOptionField(ptxField):
  '''
  A field with mutually exclusive options (or that may be None!)
  '''
  def __init__(self, name, values, ptxstrs, default = None):
    self.values = values
    self.ptxstrs = ptxstrs
    ptxField.__init__(self, name, default)

  def check(self, value):
    return value in self.values or value == None

  def render(self, value):
    if value in self.values:
      return self.ptxstrs[self.values.index(value)]
    return ''

  def __eq__(self, other):
    return type(self) == type(other) and self.name == other.name

class ptxOperandField(ptxField):
  def check(self, value):
    return isinstance(value, (regs.ptxVariable, spe.Variable)) or isinstance(value, (int, long, float))

  def render(self, value):
    if isinstance(value, regs.ptxVariable):
      return value.render()
    elif isinstance(value, spe.Variable):
      return value.reg.render()
    else:
      return str(value)

class ptxPredicateField(ptxField):
  def check(self, value):
    return (isinstance(value, regs.ptxVariable) and value.type == 'pred') or \
           (isinstance(value, speVariable) and value.reg.type == 'pred')

  def render(self, value):
    if isinstance(value, regs.ptxVariable):
      return value.render()
    elif isinstance(value, spe.Variable):
      return value.reg.render()

class ptxOperandOrAddressOrLabelField(ptxField):
  def check(self, value):
    return isinstance(value, (regs.ptxVariable, spe.Variable)) or isinstance(value, regs.ptxAddress) or isinstance(value, Label) or isinstance(value, (float, int, long))

  def render(self, value):
    if isinstance(value, regs.ptxVariable):
      return value.render()
    elif isinstance(value, spe.Variable):
      return value.reg.render()
    elif isinstance(value, Label):
      return value.name
    else:
      return str(value)

class ptxRegisterOrLabelField(ptxField):
  def check(self, value):
    return (isinstance(value, regs.ptxRegister) or isinstance(value, Label)) or \
           (isinstance(value, spe.Variable) and isinstance(value.reg, regs.ptxRegister))

  def render(self, value):
    if isinstance(value, regs.ptxVariable):
      return value.render()
    elif isinstance(value, spe.Variable):
      return value.reg.render()
    elif isinstance(value, Label):
      return value.name
    else:
      return str(value)

class ptxAddressField(ptxField):
  def check(self, value):
    return isinstance(value, regs.ptxAddress)

  def render(self, value):
    return value.render()

class ptxImmediateField(ptxField):
  def check(self, value):
    # TODO: Real range checking for barrier number
    return isinstance(value, (int, long)) and (value < 16)

  def render(self, value):
    return str(value)

sat = ptxFlagField('sat', '.sat', False)
cc = ptxFlagField('cc', '.cc', False)
wide = ptxFlagField('wide', '.wide', False)
uni = ptxFlagField('uni', '.uni', False)
rnd = ptxOptionField('rnd', ['rn', 'rz', 'rm', 'rp'], ['.rn', '.rz', '.rm', '.rp'], 'rn')
hlw = ptxOptionField('hlw', ['hi', 'lo', 'wide'], ['.hi', '.lo', '.wide'], 'lo')
hl = ptxOptionField('hl', ['hi', 'lo'], ['.hi', '.lo'], 'lo')
d = ptxOperandField('d')
s0 = ptxOperandField('s0')
s1 = ptxOperandField('s1')
s2 = ptxOperandField('s2')
s_or_a = ptxOperandOrAddressOrLabelField('s_or_a')
r_or_l = ptxRegisterOrLabelField('r_or_l')
a = ptxAddressField('a')
pred = ptxPredicateField('pred', None)
npred = ptxPredicateField('npred', None)
volatile = ptxFlagField('volatile', '.volatile', False)
space = ptxOptionField('space', ['const', 'global', 'local', 'param', 'shared', 'sreg'], ['.const', '.global', '.local', '.param', '.shared', '.sreg'])
imm = ptxImmediateField('imm')
compop = ptxOptionField('compop',
                        ['eq', 'ne', 'lt', 'le', 'gt', 'ge', 'lo', 'ls', 'hi', 'hs', 'equ', 'neu', 'ltu', 'leu', 'gtu', 'geu', 'num', 'nan'],
                        ['.eq', '.ne', '.lt', '.le', '.gt', '.ge', '.lo', '.ls', '.hi', '.hs', '.equ', '.neu', '.ltu', '.leu', '.gtu', '.geu', '.num', '.nan'])
boolop = ptxOptionField('boolop', ['and', 'or', 'xor'], ['.and', '.or', '.xor'])
redop = ptxOptionField('redop',
                       ['and', 'or', 'xor', 'cas', 'exch', 'add', 'inc', 'dec', 'min', 'max'],
                       ['.and', '.or', '.xor', '.cas', '.exch', '.add', '.inc', '.dec', '.min', '.max'])
mode = ptxOptionField('mode', ['all', 'any', 'uni'], ['.all', '.any', '.uni'])
neg = ptxFlagField('neg', '!', False)
