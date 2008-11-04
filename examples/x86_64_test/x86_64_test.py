# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
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

import array
import subprocess
import sys

import corepy.arch.x86_64.isa as x86
from corepy.arch.x86_64.types.registers import *
import corepy.arch.x86_64.platform as env
from corepy.arch.x86_64.lib.memory import MemRef
import corepy.lib.printer as printer


def get_nasm_output(code, inst):
  """Take an instruction, and return a hex string of its encoding, as encoded by GAS"""
 
  fd = open("x86_64_test.s", "w")
  printer.PrintInstructionStream(code, printer.x86_64_Nasm(function_name="_start"), fd = fd)
  fd.close()

  ret = subprocess.call(["nasm", "-Ox", "x86_64_test.s"])
  if ret != 0:
    return

  output = subprocess.Popen(["xxd", "-ps", "x86_64_test"], stdout=subprocess.PIPE).communicate()[0]
  hex = ''.join(output.splitlines())

  # If the prolog/epilog change, these need to be updated
  #startstr = "554889e54157415641554154575653"
  startstr = "554889e5415741564155415453"
  #stopstr = "5b5e5f415c415d415e415fc9c3"
  stopstr = "5b415c415d415e415fc9c3"
  startpos = hex.find(startstr) + len(startstr)
  stoppos = hex.find(stopstr)

  return hex[startpos:stoppos]


def get_corepy_output(code, inst):
  """Take an instruction, and return a hex string of its encoding, as encoded by CorePy"""
  hex_list = inst.render()

  hex = ""
  for x in hex_list:
    hex += "%02x" % (x)
  return hex


def ops_from_sig(code, sig):
  ops = []

  for s in sig:
    if isinstance(s, x86.x86RegisterOperand):
      if s == x86.reg64_t:
        ops.append(r12)
      elif s == x86.reg32_t:
        ops.append(edx)
      elif s == x86.reg16_t:
        ops.append(cx)
      elif s == x86.reg8_t:
        ops.append(bl)
      elif s == x86.regst_t:
        ops.append(st1)
      elif s == x86.mmx_t:
        ops.append(mm3)
      elif s == x86.xmm_t:
        ops.append(xmm5)
    elif isinstance(s, x86.FixedRegisterOperand):
      ops.append(globals()[s.name])
    elif isinstance(s, x86.x86ConstantOperand):
      ops.append(s.const)
    elif isinstance(s, x86.x86MemoryOperand):
      if s == x86.mem128_t:
        ops.append(MemRef(r9, -16, data_size = 128))
      elif s == x86.mem64_t:
        ops.append(MemRef(r12, 32, data_size = 64))
      elif s == x86.mem32_t:
        ops.append(MemRef(r15, 1024, data_size = 32))
      elif s == x86.mem16_t:
        ops.append(MemRef(rsi, data_size = 16))
      elif s == x86.mem8_t:
        ops.append(MemRef(rbp, -8, data_size = 8))
      elif s == x86.mem80_t:
        ops.append(MemRef(rdi, data_size = 80))
      elif s == x86.mem_t:
        ops.append(MemRef(rbx, data_size = None))
      else:
        ops.append(MemRef(rsp, data_size = s.size))
    elif isinstance(s, x86.Imm8):
      ops.append(13)
    elif isinstance(s, x86.Imm16):
      ops.append(10234)
    elif isinstance(s, x86.Imm32):
      ops.append(0x1EADBEEF)
    elif isinstance(s, x86.Imm64):
      ops.append(0x1EADDEADBEEF)
    elif isinstance(s, x86.Rel8off):
      ops.append(4)
    #elif isinstance(s, x86.Rel16off):
    #  ops.append(260)
    elif isinstance(s, x86.Rel32off):
      ops.append(65541)
    elif isinstance(s, x86.x86ImmediateOperand):
      ops.append(21)
    elif isinstance(s, x86.x86LabelOperand):
      ops.append(code.lbl_body)
    else:
      raise Exception("unhandled operand %s" % str(s))

  return ops


def test_inst(code, inst):
  code.add(inst)
  code.cache_code()

  nasm_hex_str = get_nasm_output(code, inst)
  corepy_hex_str = get_corepy_output(code, inst)

  if nasm_hex_str == None:
      print "***************************  NASM ERROR"
      print "corepy output:", corepy_hex_str
      printer.PrintInstructionStream(code,
          printer.x86_64_Nasm(show_epilogue = False, show_prologue = False))
      return 'nasm_fail'
  elif nasm_hex_str == corepy_hex_str:
    print "PASS"
    return 'pass'
  else:
    nasm_rex = int(nasm_hex_str[0:2], 16)
    corepy_rex = int(corepy_hex_str[0:2], 16)
    if corepy_rex - nasm_rex == 8 and (nasm_rex & 0xF0 == 0x40):
      print "WARNING CorePy is enabling 64bit for this inst, NASM is not"
      print "nasm output:   ", nasm_hex_str
      print "corepy output: ", corepy_hex_str
      return 'rex_pass'
    else:
      print "***************************  ERROR"
      print "nasm output:   ", nasm_hex_str
      print "corepy output: ", corepy_hex_str
      printer.PrintInstructionStream(code,
          printer.x86_64_Nasm(show_epilogue = False, show_prologue = False))
      return 'fail'
  return




# TODO - would like to be able to test multiple values for an operand.  ie regs
# that exercise REX differently, and forward/backward labels
# how would this be done?

if __name__ == '__main__':
  results = {'pass':0, 'rex_pass':0, 'nasm_fail':0, 'fail':0}

  #classes = [getattr(x86, cls) for cls in dir(x86) if isinstance(getattr(x86, cls), type) and issubclass(getattr(x86, cls), (x86.x86DispatchInstruction, x86.x86Instruction))]
  classes = []
  for obj in dir(x86):
    cls = getattr(x86, obj)
    if isinstance(cls, type):
      if issubclass(cls, (x86.x86DispatchInstruction, x86.x86Instruction)):
        if cls != x86.x86DispatchInstruction and cls != x86.x86Instruction:
          classes.append(cls)

  code = env.InstructionStream()
  for c in classes:
    if c == x86.int_3:
      # No way to write 'int 3' for NASM since it clashes with 'int 3' (heh)
      # So just make sure it gets rendered as 0xCC and call it a day
      inst = x86.int_3()
      code.add(inst)
      corepy_hex_str = get_corepy_output(code, inst)
      if corepy_hex_str == 'cc':
        print "PASS"
        results['pass'] += 1
      else:
        print "***************************  ERROR"
        print "corepy output:", corepy_hex_str
        results['pass'] += 1
    elif issubclass(c, x86.x86DispatchInstruction):
      for d in c.dispatch:
        code.reset()

        ops = ops_from_sig(code, d[0].signature)
        inst = c(*ops)

        print "Testing instruction:", inst

        r = test_inst(code, inst)
        results[r] += 1
        sys.stdout.flush()
        sys.stderr.flush()
    elif issubclass(c, x86.x86Instruction):
      code.reset()
      ops = ops_from_sig(code, c.machine_inst.signature)
      inst = c(*ops)

      print "Testing instruction:", inst

      r = test_inst(code, inst)
      results[r] += 1
      sys.stdout.flush()
      sys.stderr.flush()


  print "%d passes %d rex_passes" % (results['pass'], results['rex_pass'])
  print "%d failures %d NASM failures" % (results['fail'], results['nasm_fail'])
  print "%d total" % (results['pass'] + results['rex_pass'] + results['nasm_fail'] + results['fail'])

