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

import corepy.arch.x86_64.isa as x86_isa
from corepy.arch.x86_64.types.registers import *
#import corepy.arch.sse.isa as sse_isa
#from corepy.arch.sse.types import *

import corepy.arch.x86_64.platform as env
from corepy.arch.x86_64.lib.memory import MemRef

def Test():
    code = env.InstructionStream()
    proc = env.Processor()
    params = env.ExecParams()
    params.p1 = 3
    mr32 = MemRef(rbp, 16, data_size = 32)
    mr8 = MemRef(rbp, 16, data_size = 8)


    lbl1 = code.get_label("lbl1")
    lbl2 = code.get_label("lbl2")

    code.add(x86_isa.xor(rax, rax))

    code.add(x86_isa.cmp(rax, 1))
    code.add(x86_isa.jne(lbl1))

    code.add(x86_isa.ud2())
    code.add(x86_isa.ud2())

    code.add(lbl1)
    code.add(x86_isa.cmp(rax, 1))
    code.add(x86_isa.je(lbl2))
    code.add(x86_isa.add(rax, 12))
    code.add(lbl2)
    
    ret = proc.execute(code)
    print "ret", ret
    assert(ret == 12)


    code.reset()

    code.add(x86_isa.xor(rax, rax))

    code.add(x86_isa.cmp(rax, 1))
    code.add(x86_isa.jne(32))

    code.add(x86_isa.ud2())
    code.add(x86_isa.ud2())

    code.add(x86_isa.cmp(eax, 1))
    code.add(x86_isa.je(41))
    code.add(x86_isa.add(rax, 12))
   
    code.print_code(hex = True, pro = True, epi = True) 
    ret = proc.execute(code)
    print "ret", ret
    assert(ret == 12)


    code.reset()

    call_lbl = code.get_label("call_fn")

    code.add(x86_isa.xor(rax, rax))
    code.add(x86_isa.call(call_lbl))
    code.add(x86_isa.jmp(code.lbl_epilogue))
    code.add(x86_isa.mov(rax, 75))
    code.add(x86_isa.mov(rax, 42))
    code.add(call_lbl)
    code.add(x86_isa.mov(rax, 15))
    code.add(x86_isa.ret())

    code.print_code()
    ret = proc.execute(code)
    print "ret", ret
    assert(ret == 15)


    code.reset()

    fwd_lbl = code.get_label("FORWARD")
    bck_lbl = code.get_label("BACKWARD")

    code.add(x86_isa.xor(rax, rax))
    code.add(bck_lbl)
    code.add(x86_isa.cmp(rax, 1))
    code.add(x86_isa.jne(fwd_lbl))
    for i in xrange(0, 65):
      code.add(x86_isa.pop(r15))
    code.add(fwd_lbl)

    ret = proc.execute(code, mode = 'int')
    assert(ret == 0)


    code.reset()

    loop_lbl = code.get_label("LOOP")
    out_lbl = code.get_label("OUT")
    skip_lbl = code.get_label("SKIP")

    code.add(x86_isa.xor(rax, rax))
    code.add(loop_lbl)
    for i in range(0, 1):
      for i in xrange(0, 24):
        code.add(x86_isa.add(r15, MemRef(rsp, 4)))

      code.add(x86_isa.add(rax, 4))
      code.add(x86_isa.cmp(rax, 20))
      code.add(x86_isa.je(out_lbl))

      for i in xrange(0, 24):
        code.add(x86_isa.add(r15, MemRef(rsp, 4)))

      code.add(x86_isa.cmp(rax, 32))
      code.add(x86_isa.jne(loop_lbl))

    code.add(out_lbl)

    code.add(x86_isa.jmp(skip_lbl))
    for i in xrange(0, 2):
      code.add(x86_isa.add(r15, MemRef(rsp, 4)))
    code.add(skip_lbl)

    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 20)


    code.reset()

    loop_lbl = code.get_label("LOOP")
    else_lbl = code.get_label("ELSE")
    finish_lbl = code.get_label("finish")

    code.add(x86_isa.mov(rax, 0))
    code.add(x86_isa.mov(rdx, 0))

    code.add(loop_lbl)

    code.add(x86_isa.add(rax, 1))
    code.add(x86_isa.cmp(rax, 16))
    code.add(x86_isa.jge(finish_lbl))

    code.add(x86_isa.add(rdx, rax))
    code.add(x86_isa.mov(r8, rdx))
    code.add(x86_isa.and_(r8, 0x1))
    code.add(x86_isa.jnz(else_lbl))

    code.add(x86_isa.add(rdx, 1))
    code.add(x86_isa.jmp(loop_lbl))

    code.add(else_lbl)
    code.add(x86_isa.add(rdx, r8))
    code.add(x86_isa.jmp(loop_lbl))

    code.add(finish_lbl)
    code.add(x86_isa.mov(rax, rdx))

    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 135)


    code.reset()

    loop_lbl = code.get_label("LOOP")

    code.add(x86_isa.xor(rax, rax))
    code.add(x86_isa.xor(rcx, rcx))
    code.add(x86_isa.mov(rdx, 1))

    code.add(loop_lbl)
    code.add(x86_isa.inc(rax))
    code.add(x86_isa.cmp(rax, 7))
    code.add(x86_isa.cmove(rcx, rdx))
    code.add(x86_isa.jrcxz(loop_lbl))

    code.print_code(hex = True)
    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 7)


    code.reset()

    code.add(x86_isa.mov(rax, MemRef(rbp, 16)))
    code.add(x86_isa.xor(rbx, rbx))
    code.add(x86_isa.mov(rbx, -1))
    code.add(x86_isa.mov(cl, 1))
    code.add(x86_isa.shld(rax,rbx,cl))
    ret = proc.execute(code, params = params, mode = 'int')
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
    code.add(x86_isa.imul(eax, ebx, 10))
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
    assert(ret == 1280)
    ret = proc.join(id1)
    print "Return thread 1: %d" % (ret)
    assert(ret == 1280)
    ret = proc.join(id2)
    print "Return thread 2: %d" % (ret)
    assert(ret == 1280)


    code.reset()

    code.add(x86_isa.fldpi())
    code.add(x86_isa.pxor(xmm0, xmm0))
    code.add(x86_isa.fld1())
    code.add(x86_isa.fadd(st0, st0))
    code.add(x86_isa.fmulp())
    code.add(x86_isa.fsin())
    code.add(x86_isa.fcos())
    code.add(x86_isa.fld1())
    code.add(x86_isa.fyl2xp1())

    # x86_64 now uses xmm0 to return floats, not st0.  So here, just make room
    # on the stack, convert the FP result to an int and store it on the stack,
    # then pop it into rax, the int return register.
    code.add(x86_isa.push(rax))
    code.add(x86_isa.fistp(MemRef(rsp)))
    code.add(x86_isa.pop(rax))

    code.print_code(hex = True)
    ret = proc.execute(code, params = params, mode = 'int')
    assert(ret == 1)
    print "Return main thread: %d" % (ret)


    code.reset()

    lbl_ok = code.get_label("OK")
    code.add(x86_isa.emms())
    code.add(x86_isa.movd(xmm0, mr32))
    code.add(x86_isa.mov(ebx, mr32))

    code.add(x86_isa.cmp(ebx, 3))
    code.add(x86_isa.je(lbl_ok))
    code.add(x86_isa.movd(eax, xmm0))
    code.add(x86_isa.cmp(eax, 3))
    code.add(x86_isa.je(lbl_ok))
    code.add(x86_isa.ud2())

    code.add(lbl_ok)
    code.add(x86_isa.xor(eax, eax))
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
    assert(ret == 6)


    code.reset()

    # Test immediate size encodings
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
    #code.print_code(hex = True)

    proc.execute(code, params = params)


    code.reset()

    code.add(x86_isa.mov(edx, 0x1234))
    code.add(x86_isa.mov(eax, 0xFFFF))
    code.add(x86_isa.xchg(edx, eax))

    code.print_code(hex = True)
    ret = proc.execute(code, params = params)
    print "ret:", ret
    assert(ret == 0x1234)


    code.reset()

    code.add(x86_isa.mov(rax, rsp))
    code.add(x86_isa.pushfq())
    code.add(x86_isa.sub(rax, rsp))
    code.add(x86_isa.add(rsp, rax))

    code.print_code(hex = True)
    ret = proc.execute(code, params = params)
    print "ret:", ret
    assert(ret == 8)

    return

Test()

