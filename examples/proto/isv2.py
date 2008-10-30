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

# Prototypes for InstructionStream Version 2

# Goals:
#   o Separation of InstructionStream and BinaryInstructionStream
#     - Same idea as Instruction/MachineInstruction
#     - IS has all user level IS operations
#     - BIS is the machine specific binary rendering
#   o Modular instruction streams
#     - Somewhere between basic blocks and functions
#     - Can be used to implement any type of code block abstraction  
#   o Composable instruction streams
#     - Joins, branches
#     - Interleaving, pipelining
#   o Branch/label abstractions (reinvent goto!)
#   o Register abstractions
#     - Register 'ports'
#     - Named registers
#   o Procedural features
#     - optional stack (for IA-32)
#     - link register

# Ambitious Goals:
#   o Capture advanced programming contructs easily
#     - Continuations
#     - PIM
#     - etc

# Tanimoto example

# This is an intersting idea... objects that when constructed are ISs
# for the function.  Kinda like the ideas used for synthesis, but more
# self contained.
div = DivIS()
tan = Tanimoto()

block_proc = BlockProc(kernel = Tanimoto, x_addr = Register_Port, y_addr = Register_Port)


code = IS2()
x_addr = code.named_register('x_addr')
y_addr = code.named_register('y_addr')

data = make_array()
x_code = DataStreamIS(data)

block_proc.set_x_stream(x_code)
