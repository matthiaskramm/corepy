
# import platform_conf
import vmx_isa as machine
import corepy.spre.spe as spe

# Nothing to see here, move along... ;)
__active_code = None

def set_active_code(code):
  global __active_code

  if __active_code is not None:
    __active_code.set_active_callback(None)

  __active_code = code

  if code is not None:
    code.set_active_callback(set_active_code)
  return

# Property version
def __get_active_code(self):
  global __active_code
  return __active_code

# Free function version
def get_active_code():
  global __active_code
  return __active_code

# Build the instructions
for inst in vmx_isa.VMX_ISA:
  name = inst[0]
  machine_inst = getattr(machine, name)
  
  # asm_order = inst[1]['asm']
  members = {}
  for key in inst[1].keys():
    members[key] = inst[1][key]

  members['asm_order'] =  members['asm']
  members['machine_inst'] =  machine_inst
  members['active_code']  = property(__get_active_code) 
  globals()[inst[0]] = type(name, (spe.Instruction,), members)
                                                       
# Copyright 2006 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

