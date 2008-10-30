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

import corepy.arch.x86.isa as x86_isa
from corepy.arch.x86.types.registers import *
#import corepy.arch.sse.isa as sse_isa
#from corepy.arch.sse.types import *

import corepy.arch.x86.platform as env
from corepy.arch.x86.lib.memory import MemRef

def Test():
    code = env.InstructionStream()
    proc = env.Processor()
    params = env.ExecParams()
    params.p1 = 3
    mr32 = MemRef(ebp, 8)
    mr8 = MemRef(ebp, 8, data_size = 8)


    lbl1 = code.get_label("lbl1")
    lbl2 = code.get_label("lbl2")

    code.add(x86_isa.xor(eax, eax))

    code.add(x86_isa.cmp(eax, 1))
    code.add(x86_isa.jne(lbl1))

    code.add(x86_isa.ret())

    code.add(lbl1)
    code.add(x86_isa.cmp(eax, 1))
    code.add(x86_isa.je(lbl2))
    code.add(x86_isa.add(eax, 12))
    code.add(lbl2)
    
    ret = proc.execute(code)
    print "ret", ret
    assert(ret == 12)


    code.reset()

    fwd_lbl = code.get_label("FORWARD")
    bck_lbl = code.get_label("BACKWARD")

    code.add(x86_isa.xor(eax, eax))
    code.add(bck_lbl)
    code.add(x86_isa.cmp(eax, 1))
    code.add(x86_isa.jne(fwd_lbl))
    for i in xrange(0, 65):
      code.add(x86_isa.pop(edi))
    code.add(fwd_lbl)

    ret = proc.execute(code, mode = 'int')
    assert(ret == 0)


    code.reset()

    loop_lbl = code.get_label("LOOP")
    out_lbl = code.get_label("OUT")
    skip_lbl = code.get_label("SKIP")

    code.add(x86_isa.xor(eax, eax))
    code.add(loop_lbl)
    for i in range(0, 1):
      for i in xrange(0, 24):
        code.add(x86_isa.add(edx, MemRef(esp, 4)))

      code.add(x86_isa.add(eax, 4))
      code.add(x86_isa.cmp(eax, 20))
      code.add(x86_isa.je(out_lbl))

      for i in xrange(0, 24):
        code.add(x86_isa.add(edx, MemRef(esp, 4)))

      code.add(x86_isa.cmp(eax, 32))
      code.add(x86_isa.jne(loop_lbl))

    code.add(out_lbl)

    code.add(x86_isa.jmp(skip_lbl))
    for i in xrange(0, 2):
      code.add(x86_isa.add(edx, MemRef(esp, 4)))
    code.add(skip_lbl)

    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 20)


    code.reset()

    loop_lbl = code.get_label("LOOP")
    else_lbl = code.get_label("ELSE")
    finish_lbl = code.get_label("finish")

    code.add(x86_isa.mov(eax, -1))
    code.add(x86_isa.mov(edx, 0))

    code.add(loop_lbl)

    code.add(x86_isa.add(eax, 1))
    code.add(x86_isa.cmp(eax, 16))
    code.add(x86_isa.jge(finish_lbl))

    code.add(x86_isa.add(edx, eax))
    code.add(x86_isa.mov(edx, edi))
    code.add(x86_isa.and_(edi, 0x1))
    code.add(x86_isa.jnz(else_lbl))

    code.add(x86_isa.add(edx, 1))
    code.add(x86_isa.jmp(loop_lbl))

    code.add(else_lbl)
    code.add(x86_isa.add(edx, edi))
    code.add(x86_isa.jmp(loop_lbl))

    code.add(finish_lbl)
    code.add(x86_isa.mov(eax, edx))

    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 1)


    code.reset()

    loop_lbl = code.get_label("LOOP")

    code.add(x86_isa.xor(eax, eax))
    code.add(x86_isa.xor(ecx, ecx))
    code.add(x86_isa.mov(edx, 1))

    code.add(loop_lbl)
    code.add(x86_isa.inc(eax))
    code.add(x86_isa.cmp(eax, 7))
    code.add(x86_isa.cmove(ecx, edx))
    code.add(x86_isa.jecxz(loop_lbl))

    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 7)


    code.reset()

    code.add(x86_isa.add(eax, 200))
    code.add(x86_isa.xor(eax, eax))
    code.add(x86_isa.add(al, 32))
    code.add(x86_isa.add(bl, 32))
    code.add(x86_isa.xor(bl, bl))
    code.add(x86_isa.mov(mr8, al))
    code.add(x86_isa.add(mr32, 0))
    code.add(x86_isa.mov(eax, mr32))
    code.add(x86_isa.mov(al, mr8))

    code.add(x86_isa.imul(ax, ax, 4))
    code.add(x86_isa.imul(ebx, eax, 10))
    code.add(x86_isa.mov(cx, 1232))
    code.add(x86_isa.sub(ax, cx))
    code.add(x86_isa.xor(eax,eax))
    code.add(x86_isa.mov(eax,ebx))
    code.add(x86_isa.clc())
    code.add(x86_isa.rcl(eax, 1))
    code.add(x86_isa.rcr(eax, 1))


    #ret = proc.execute(code, debug = True, params = params)
    id1 = proc.execute(code, params = params, mode = 'int', async = True)
    id2 = proc.execute(code, params = params, mode = 'int', async = True)
    ret = proc.execute(code, params = params, mode = 'int')
    print "Return main thread: %d" % (ret)
    ret = proc.join(id1)
    print "Return thread 1: %d" % (ret)
    ret = proc.join(id2)
    print "Return thread 2: %d" % (ret)

    code.reset()
    code.add(x86_isa.fldpi())
    code.add(x86_isa.fld1())
    code.add(x86_isa.fadd(st0, st0))
    code.add(x86_isa.fmulp())
    code.add(x86_isa.fsin())
    code.add(x86_isa.fcos())
    code.add(x86_isa.fld1())
    code.add(x86_isa.fyl2xp1())

    ret = proc.execute(code, params = params, mode = 'fp')
    print "Return main thread: %f" % (ret)

    code.reset()
    code.add(x86_isa.movd(xmm0, mr32))
    code.add(x86_isa.mov(ebx, mr32))
    code.add(x86_isa.movd(xmm1, ebx))
    code.add(x86_isa.paddq(xmm0, xmm1))
    code.add(x86_isa.pextrw(ecx, xmm0, 0))
    code.add(x86_isa.pinsrw(mm1, ecx, 0))
    code.add(x86_isa.movq2dq(xmm0, mm1))
    code.add(x86_isa.movdq2q(mm2, xmm0))
    code.add(x86_isa.movd(edx,mm2))
    code.add(x86_isa.movd(xmm5,edx))
    code.add(x86_isa.movd(ecx, xmm5))
    code.add(x86_isa.pinsrw(xmm6, ecx, 0))
    code.add(x86_isa.movd(eax, xmm6))

    code.print_code(hex = True)
    ret = proc.execute(code, params = params, mode = 'int')
    print "Return main thread: %d" % (ret)

    # Test immediate size encodings
    code.reset()
    code.add(x86_isa.add(eax, 300))
    code.add(x86_isa.add(ax, 300))
    code.add(x86_isa.add(ax, 30))
    code.add(x86_isa.mov(eax, 16))
    code.add(x86_isa.mov(eax, 300))

    code.reset()
    code.add(x86_isa.add(eax, 0xDEADBEEF))
    code.add(x86_isa.add(ebx, 0xDEADBEEF))
    code.print_code(hex = True)

    # Try the LOCK prefix
    code.reset()
    code.add(x86_isa.xor(eax, eax))
    code.add(x86_isa.add(mr32, eax))
    code.add(x86_isa.add(mr32, eax, lock = True))
    code.print_code(hex = True)

    proc.execute(code, params = params)
    return

Test()

