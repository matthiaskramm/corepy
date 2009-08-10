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

import array

import corepy.arch.x86_64.isa as x86
from corepy.arch.x86_64.types.registers import *
import corepy.arch.x86_64.platform as env
from corepy.arch.x86_64.lib.memory import MemRef
import corepy.lib.extarray as extarray
import corepy.arch.x86_64.lib.util as util

def Test():
    prgm = env.Program()
    code = prgm.get_stream()
    proc = env.Processor()
    params = env.ExecParams()
    params.p1 = 3

    lbl1 = prgm.get_label("lbl1")
    lbl2 = prgm.get_label("lbl2")

    code.add(x86.xor(prgm.gp_return, prgm.gp_return))

    code.add(x86.cmp(prgm.gp_return, 1))
    code.add(x86.jne(lbl1))

    code.add(x86.ud2())
    code.add(x86.ud2())

    code.add(lbl1)
    code.add(x86.cmp(prgm.gp_return, 1))
    code.add(x86.je(lbl2))
    code.add(x86.add(prgm.gp_return, 12))
    code.add(lbl2)

    prgm.add(code)
    #prgm.print_code(pro = True, epi = True, hex = True) 
    ret = proc.execute(prgm, mode = 'int')
    print "ret", ret
    assert(ret == 12)

    prgm.reset()
    code.reset()

    code.add(x86.xor(prgm.gp_return, prgm.gp_return))

    code.add(x86.cmp(prgm.gp_return, 1))
    code.add(x86.jne(28))

    code.add(x86.ud2())
    code.add(x86.ud2())

    code.add(x86.cmp(prgm.gp_return, 1))
    code.add(x86.je(37))
    code.add(x86.add(prgm.gp_return, 12))

    prgm.add(code)
    prgm.print_code(hex = True, pro = True, epi = True) 
    ret = proc.execute(prgm)
    print "ret", ret
    assert(ret == 12)

    prgm.reset()
    code.reset()

    call_lbl = prgm.get_label("call_fn")

    code.add(x86.xor(prgm.gp_return, prgm.gp_return))
    code.add(x86.call(call_lbl))
    code.add(x86.jmp(prgm.lbl_epilogue))
    code.add(x86.mov(prgm.gp_return, 75))
    code.add(x86.mov(prgm.gp_return, 42))
    code.add(call_lbl)
    code.add(x86.mov(prgm.gp_return, 15))
    code.add(x86.ret())

    prgm.add(code)
    prgm.print_code()
    ret = proc.execute(prgm, mode = 'int')
    print "ret", ret
    assert(ret == 15)


    prgm.reset()
    code.reset()

    fwd_lbl = prgm.get_label("FORWARD")
    bck_lbl = prgm.get_label("BACKWARD")

    code.add(x86.xor(prgm.gp_return, prgm.gp_return))
    code.add(bck_lbl)
    code.add(x86.cmp(prgm.gp_return, 1))
    code.add(x86.jne(fwd_lbl))
    r_foo = prgm.acquire_register()
    for i in xrange(0, 65):
      code.add(x86.pop(r_foo))
    prgm.release_register(r_foo)
    code.add(fwd_lbl)

    prgm.add(code)
    ret = proc.execute(prgm, mode = 'int')
    print "ret", ret
    assert(ret == 0)


    prgm.reset()
    code.reset()

    loop_lbl = prgm.get_label("LOOP")
    out_lbl = prgm.get_label("OUT")
    skip_lbl = prgm.get_label("SKIP")

    code.add(x86.xor(prgm.gp_return, prgm.gp_return))
    code.add(loop_lbl)
    r_foo = prgm.acquire_register()
    for i in range(0, 1):
      for i in xrange(0, 24):
        code.add(x86.add(r_foo, MemRef(rsp, 4)))

      code.add(x86.add(prgm.gp_return, 4))
      code.add(x86.cmp(prgm.gp_return, 20))
      code.add(x86.je(out_lbl))

      for i in xrange(0, 24):
        code.add(x86.add(r_foo, MemRef(rsp, 4)))

      code.add(x86.cmp(prgm.gp_return, 32))
      code.add(x86.jne(loop_lbl))

    code.add(out_lbl)

    code.add(x86.jmp(skip_lbl))
    for i in xrange(0, 2):
      code.add(x86.add(r_foo, MemRef(rsp, 4)))
    code.add(skip_lbl)

    prgm.release_register(r_foo)
    prgm.add(code)
    ret = proc.execute(prgm, mode = 'int')
    print "ret", ret
    assert(ret == 20)


    prgm.reset()
    code.reset()

    r_tmp = prgm.acquire_registers(2)

    loop_lbl = prgm.get_label("LOOP")
    else_lbl = prgm.get_label("ELSE")
    finish_lbl = prgm.get_label("finish")

    code.add(x86.mov(prgm.gp_return, 0))
    code.add(x86.mov(r_tmp[0], 0))

    code.add(loop_lbl)

    code.add(x86.add(prgm.gp_return, 1))
    code.add(x86.cmp(prgm.gp_return, 16))
    code.add(x86.jge(finish_lbl))

    code.add(x86.add(r_tmp[0], prgm.gp_return))
    code.add(x86.mov(r_tmp[1], r_tmp[0]))
    code.add(x86.and_(r_tmp[1], 0x1))
    code.add(x86.jnz(else_lbl))

    code.add(x86.add(r_tmp[0], 1))
    code.add(x86.jmp(loop_lbl))

    code.add(else_lbl)
    code.add(x86.add(r_tmp[0], r_tmp[1]))
    code.add(x86.jmp(loop_lbl))

    code.add(finish_lbl)
    code.add(x86.mov(prgm.gp_return, r_tmp[0]))

    prgm.release_registers(r_tmp)

    prgm.add(code)
    ret = proc.execute(prgm, mode = 'int')
    print "ret", ret
    assert(ret == 135)


    prgm.reset()
    code.reset()

    loop_lbl = prgm.get_label("LOOP")

    r_one = prgm.acquire_register()
    code.add(x86.xor(prgm.gp_return, prgm.gp_return))
    code.add(x86.xor(rcx, rcx))
    code.add(x86.mov(r_one, 1))

    code.add(loop_lbl)
    code.add(x86.inc(prgm.gp_return))
    code.add(x86.cmp(prgm.gp_return, 7))
    code.add(x86.cmove(rcx, r_one))
    code.add(x86.jrcxz(loop_lbl))

    prgm.release_register(r_one)

    prgm.add(code)
    prgm.print_code(hex = True)
    ret = proc.execute(prgm, mode = 'int')
    print "ret", ret
    assert(ret == 7)


    prgm.reset()
    code.reset()

    r_tmp = prgm.acquire_register()
    code.add(x86.mov(prgm.gp_return, rdi))
    code.add(x86.xor(r_tmp, r_tmp))
    code.add(x86.mov(r_tmp, -1))
    code.add(x86.mov(cl, 1))
    code.add(x86.shld(prgm.gp_return, r_tmp, cl))

    prgm.release_register(r_tmp)
    prgm.add(code)
    ret = proc.execute(prgm, params = params, mode = 'int')
    print "ret", ret
    assert(ret == 7)


    prgm.reset()
    code.reset()

    code.add(x86.add(eax, 200))
    code.add(x86.xor(eax, eax))
    code.add(x86.add(al, 32))
    code.add(x86.add(bl, 32))
    code.add(x86.xor(bl, bl))
    code.add(x86.mov(dil, al))
    code.add(x86.add(rdi, 0))
    code.add(x86.mov(eax, edi))
    code.add(x86.mov(al, dil))

    code.add(x86.imul(ax, ax, 4))
    code.add(x86.imul(eax, ebx, 10))
    code.add(x86.mov(cx, 1232))
    code.add(x86.sub(ax, cx))
    code.add(x86.xor(eax,eax))
    code.add(x86.mov(eax,ebx))
    code.add(x86.clc())
    code.add(x86.rcl(eax, 1))
    code.add(x86.rcr(eax, 1))


    prgm.add(code)
    #ret = proc.execute(prgm, debug = True, params = params)
    id1 = proc.execute(prgm, params = params, mode = 'int', async = True)
    id2 = proc.execute(prgm, params = params, mode = 'int', async = True)
    ret = proc.execute(prgm, params = params, mode = 'int')
    print "Return main thread: %d" % (ret)
    assert(ret == 1280)
    ret = proc.join(id1)
    print "Return thread 1: %d" % (ret)
    assert(ret == 1280)
    ret = proc.join(id2)
    print "Return thread 2: %d" % (ret)
    assert(ret == 1280)


    prgm.reset()
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
    code.add(x86.push(prgm.gp_return))
    code.add(x86.fistp(MemRef(rsp)))
    code.add(x86.pop(prgm.gp_return))

    prgm.add(code)
    prgm.print_code(hex = True)
    ret = proc.execute(prgm, params = params, mode = 'int')
    assert(ret == 1)
    print "Return main thread: %d" % (ret)


    prgm.reset()
    code.reset()

    lbl_ok = prgm.get_label("OK")
    code.add(x86.emms())
    code.add(x86.movd(xmm0, edi))
    code.add(x86.mov(ebx, edi))

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
    code.add(x86.pxor(mm1, mm1))
    code.add(x86.pinsrw(mm1, ecx, 0))
    code.add(x86.movq2dq(xmm0, mm1))
    code.add(x86.movdq2q(mm2, xmm0))
    code.add(x86.movd(edx,mm2))
    code.add(x86.movd(xmm5,edx))
    code.add(x86.movd(ecx, xmm5))
    code.add(x86.pxor(xmm6, xmm6))
    code.add(x86.pinsrw(xmm6, ecx, 0))
    code.add(x86.movd(eax, xmm6))

    prgm.add(code)
    prgm.print_code(hex = True)
    ret = proc.execute(prgm, params = params, mode = 'int')
    print "Return main thread: %d" % (ret)
    assert(ret == 6)


    prgm.reset()
    code.reset()

    code.add(x86.mov(edx, 0x1234))
    code.add(x86.mov(eax, 0xFFFF))
    code.add(x86.xchg(edx, eax))

    prgm.add(code)
    prgm.print_code(hex = True)
    ret = proc.execute(prgm, params = params)
    print "ret:", ret
    assert(ret == 0x1234)


    prgm.reset()
    code.reset()

    code.add(x86.mov(prgm.gp_return, rsp))
    code.add(x86.pushfq())
    code.add(x86.sub(prgm.gp_return, rsp))
    code.add(x86.add(rsp, prgm.gp_return))

    prgm.add(code)
    prgm.print_code(hex = True)
    ret = proc.execute(prgm, params = params)
    print "ret:", ret
    assert(ret == 8)


    prgm.reset()
    code.reset()

    data = extarray.extarray('H', xrange(0, 16))

    r_128 = prgm.acquire_register(reg_type = XMMRegister)
    regs = prgm.acquire_registers(4)

    code.add(x86.mov(regs[0], data.buffer_info()[0]))
    code.add(x86.movaps(r_128, MemRef(regs[0], data_size = 128)))
    code.add(x86.pextrw(prgm.gp_return, r_128, 0))
    code.add(x86.pextrw(regs[1], r_128, 1))
    code.add(x86.pextrw(regs[2], r_128, 2))
    code.add(x86.pextrw(regs[3], r_128, 3))
    code.add(x86.shl(regs[1], 16))
    code.add(x86.shl(regs[2], 32))
    code.add(x86.shl(regs[3], 48))
    code.add(x86.or_(prgm.gp_return, regs[1]))
    code.add(x86.or_(prgm.gp_return, regs[2]))
    code.add(x86.or_(prgm.gp_return, regs[3]))

    prgm.release_register(r_128)
    prgm.release_registers(regs)

    prgm.add(code)
    prgm.print_code()
    ret = proc.execute(prgm, mode = 'int')
    print "ret %x" % ret
    assert(ret == 0x0003000200010000)


    prgm.reset()
    code.reset()

    util.load_float(code, xmm0, 3.14159)

    prgm.add(code)
    ret = proc.execute(prgm, mode = 'fp')
    print "ret", ret
    assert(ret - 3.14159 < 0.00001)

    return

Test()

