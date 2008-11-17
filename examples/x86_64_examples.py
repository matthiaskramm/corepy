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

import corepy.arch.x86_64.isa as x86
from corepy.arch.x86_64.types.registers import *
import corepy.arch.x86_64.platform as env
from corepy.arch.x86_64.lib.memory import MemRef
import corepy.lib.extarray as extarray

def Test():
    code = env.InstructionStream()
    proc = env.Processor()
    params = env.ExecParams()
    params.p1 = 3
    mr32 = MemRef(rbp, 16, data_size = 32)
    mr8 = MemRef(rbp, 16, data_size = 8)


    lbl1 = code.get_label("lbl1")
    lbl2 = code.get_label("lbl2")

    code.add(x86.xor(rax, rax))

    code.add(x86.cmp(rax, 1))
    code.add(x86.jne(lbl1))

    code.add(x86.ud2())
    code.add(x86.ud2())

    code.add(lbl1)
    code.add(x86.cmp(rax, 1))
    code.add(x86.je(lbl2))
    code.add(x86.add(rax, 12))
    code.add(lbl2)
    
    ret = proc.execute(code)
    print "ret", ret
    assert(ret == 12)

    print "W00T"

    code.reset()

    code.add(x86.xor(rax, rax))

    code.add(x86.cmp(rax, 1))
    code.add(x86.jne(28))

    code.add(x86.ud2())
    code.add(x86.ud2())

    code.add(x86.cmp(eax, 1))
    code.add(x86.je(37))
    code.add(x86.add(rax, 12))
   
    code.print_code(hex = True, pro = True, epi = True) 
    print "a"
    ret = proc.execute(code)
    print "b"
    print "ret", ret
    assert(ret == 12)

    print "w00t 2"

    code.reset()

    call_lbl = code.get_label("call_fn")

    code.add(x86.xor(rax, rax))
    code.add(x86.call(call_lbl))
    code.add(x86.jmp(code.lbl_epilogue))
    code.add(x86.mov(rax, 75))
    code.add(x86.mov(rax, 42))
    code.add(call_lbl)
    code.add(x86.mov(rax, 15))
    code.add(x86.ret())

    code.print_code()
    ret = proc.execute(code)
    print "ret", ret
    assert(ret == 15)


    code.reset()

    fwd_lbl = code.get_label("FORWARD")
    bck_lbl = code.get_label("BACKWARD")

    code.add(x86.xor(rax, rax))
    code.add(bck_lbl)
    code.add(x86.cmp(rax, 1))
    code.add(x86.jne(fwd_lbl))
    for i in xrange(0, 65):
      code.add(x86.pop(r15))
    code.add(fwd_lbl)

    ret = proc.execute(code, mode = 'int')
    assert(ret == 0)


    code.reset()

    loop_lbl = code.get_label("LOOP")
    out_lbl = code.get_label("OUT")
    skip_lbl = code.get_label("SKIP")

    code.add(x86.xor(rax, rax))
    code.add(loop_lbl)
    for i in range(0, 1):
      for i in xrange(0, 24):
        code.add(x86.add(r15, MemRef(rsp, 4)))

      code.add(x86.add(rax, 4))
      code.add(x86.cmp(rax, 20))
      code.add(x86.je(out_lbl))

      for i in xrange(0, 24):
        code.add(x86.add(r15, MemRef(rsp, 4)))

      code.add(x86.cmp(rax, 32))
      code.add(x86.jne(loop_lbl))

    code.add(out_lbl)

    code.add(x86.jmp(skip_lbl))
    for i in xrange(0, 2):
      code.add(x86.add(r15, MemRef(rsp, 4)))
    code.add(skip_lbl)

    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 20)


    code.reset()

    loop_lbl = code.get_label("LOOP")
    else_lbl = code.get_label("ELSE")
    finish_lbl = code.get_label("finish")

    code.add(x86.mov(rax, 0))
    code.add(x86.mov(rdx, 0))

    code.add(loop_lbl)

    code.add(x86.add(rax, 1))
    code.add(x86.cmp(rax, 16))
    code.add(x86.jge(finish_lbl))

    code.add(x86.add(rdx, rax))
    code.add(x86.mov(r8, rdx))
    code.add(x86.and_(r8, 0x1))
    code.add(x86.jnz(else_lbl))

    code.add(x86.add(rdx, 1))
    code.add(x86.jmp(loop_lbl))

    code.add(else_lbl)
    code.add(x86.add(rdx, r8))
    code.add(x86.jmp(loop_lbl))

    code.add(finish_lbl)
    code.add(x86.mov(rax, rdx))

    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 135)


    code.reset()

    loop_lbl = code.get_label("LOOP")

    code.add(x86.xor(rax, rax))
    code.add(x86.xor(rcx, rcx))
    code.add(x86.mov(rdx, 1))

    code.add(loop_lbl)
    code.add(x86.inc(rax))
    code.add(x86.cmp(rax, 7))
    code.add(x86.cmove(rcx, rdx))
    code.add(x86.jrcxz(loop_lbl))

    code.print_code(hex = True)
    ret = proc.execute(code, mode = 'int')
    print "ret", ret
    assert(ret == 7)


    code.reset()

    code.add(x86.mov(rax, MemRef(rbp, 16)))
    code.add(x86.xor(rbx, rbx))
    code.add(x86.mov(rbx, -1))
    code.add(x86.mov(cl, 1))
    code.add(x86.shld(rax,rbx,cl))
    ret = proc.execute(code, params = params, mode = 'int')
    print "ret", ret
    assert(ret == 7)

    code.reset()


    code.add(x86.add(eax, 200))
    code.add(x86.xor(eax, eax))
    code.add(x86.add(al, 32))
    code.add(x86.add(bl, 32))
    code.add(x86.xor(bl, bl))
    code.add(x86.mov(mr8, al))
    code.add(x86.add(mr32, 0))
    code.add(x86.mov(eax, mr32))
    code.add(x86.mov(al, mr8))

    code.add(x86.imul(ax, ax, 4))
    code.add(x86.imul(eax, ebx, 10))
    code.add(x86.mov(cx, 1232))
    code.add(x86.sub(ax, cx))
    code.add(x86.xor(eax,eax))
    code.add(x86.mov(eax,ebx))
    code.add(x86.clc())
    code.add(x86.rcl(eax, 1))
    code.add(x86.rcr(eax, 1))


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

    code.add(x86.fldpi())
    code.add(x86.pxor(xmm0, xmm0))
    code.add(x86.fld1())
    code.add(x86.fadd(st0, st0))
    code.add(x86.fmulp())
    code.add(x86.fsin())
    code.add(x86.fcos())
    code.add(x86.fld1())
    code.add(x86.fyl2xp1())

    # x86_64 now uses xmm0 to return floats, not st0.  So here, just make room
    # on the stack, convert the FP result to an int and store it on the stack,
    # then pop it into rax, the int return register.
    code.add(x86.push(rax))
    code.add(x86.fistp(MemRef(rsp)))
    code.add(x86.pop(rax))

    code.print_code(hex = True)
    ret = proc.execute(code, params = params, mode = 'int')
    assert(ret == 1)
    print "Return main thread: %d" % (ret)


    code.reset()

    lbl_ok = code.get_label("OK")
    code.add(x86.emms())
    code.add(x86.movd(xmm0, mr32))
    code.add(x86.mov(ebx, mr32))

    code.add(x86.cmp(ebx, 3))
    code.add(x86.je(lbl_ok))
    code.add(x86.movd(eax, xmm0))
    code.add(x86.cmp(eax, 3))
    code.add(x86.je(lbl_ok))
    code.add(x86.ud2())

    code.add(lbl_ok)
    code.add(x86.xor(eax, eax))
    code.add(x86.movd(xmm1, ebx))
    code.add(x86.paddq(xmm0, xmm1))
    code.add(x86.pextrw(ecx, xmm0, 0))
    code.add(x86.pinsrw(mm1, ecx, 0))
    code.add(x86.movq2dq(xmm0, mm1))
    code.add(x86.movdq2q(mm2, xmm0))
    code.add(x86.movd(edx,mm2))
    code.add(x86.movd(xmm5,edx))
    code.add(x86.movd(ecx, xmm5))
    code.add(x86.pinsrw(xmm6, ecx, 0))
    code.add(x86.movd(eax, xmm6))

    code.print_code(hex = True)
    ret = proc.execute(code, params = params, mode = 'int')
    print "Return main thread: %d" % (ret)
    assert(ret == 6)


    code.reset()

    # Test immediate size encodings
    code.add(x86.add(eax, 300))
    code.add(x86.add(ax, 300))
    code.add(x86.add(ax, 30))
    code.add(x86.mov(eax, 16))
    code.add(x86.mov(eax, 300))

    code.reset()
    code.add(x86.add(eax, 0xDEADBEEF))
    code.add(x86.add(ebx, 0xDEADBEEF))
    code.print_code(hex = True)

    # Try the LOCK prefix
    code.reset()
    code.add(x86.xor(eax, eax))
    code.add(x86.add(mr32, eax))
    code.add(x86.add(mr32, eax, lock = True))
    #code.print_code(hex = True)

    proc.execute(code, params = params)


    code.reset()

    code.add(x86.mov(edx, 0x1234))
    code.add(x86.mov(eax, 0xFFFF))
    code.add(x86.xchg(edx, eax))

    code.print_code(hex = True)
    ret = proc.execute(code, params = params)
    print "ret:", ret
    assert(ret == 0x1234)


    code.reset()

    code.add(x86.mov(rax, rsp))
    code.add(x86.pushfq())
    code.add(x86.sub(rax, rsp))
    code.add(x86.add(rsp, rax))

    code.print_code(hex = True)
    ret = proc.execute(code, params = params)
    print "ret:", ret
    assert(ret == 8)


    code.reset()

    data = extarray.extarray('H', xrange(0, 16))

    code.add(x86.mov(rdi, data.buffer_info()[0]))
    code.add(x86.movaps(xmm1, MemRef(rdi, data_size = 128)))
    code.add(x86.pextrw(rax, xmm1, 0))
    code.add(x86.pextrw(rbx, xmm1, 1))
    code.add(x86.pextrw(rcx, xmm1, 2))
    code.add(x86.pextrw(rdx, xmm1, 3))
    code.add(x86.shl(rbx, 16))
    code.add(x86.shl(rcx, 32))
    code.add(x86.shl(rdx, 48))
    code.add(x86.or_(rax, rbx))
    code.add(x86.or_(rax, rcx))
    code.add(x86.or_(rax, rdx))

    ret = proc.execute(code, mode = 'int')
    print "ret %x" % ret
    assert(ret == 0x0003000200010000)
    return

Test()

