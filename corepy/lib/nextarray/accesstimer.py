#!/usr/bin/env python

import sys

sys.path.append("/home/paul/g/fetched/corepy-1.0")

import time
import array

import corepy.arch.x86_64.platform as env
import corepy.arch.x86_64.isa as x86
import corepy.arch.x86_64.types.registers as r
from corepy.arch.x86_64.lib.memory import MemRef
from corepy.lib.nextarray import nextarray

def timecmd(name, cmd, r):
    c = 0
    total = 0
    while total < 10:
        t = time.time()
        cmd()
        d = time.time()
        total += d - t
        c += 1
    print "%s %5.1f %d" % (name, total*(2.4E9)/(r*c), c)

reps = (1<<6)

code = env.InstructionStream()
code.add(x86.nop())
def callasm():
    for i in range(reps):
        params = env.ExecParams()
        env.Processor().execute(code)
timecmd("callasm", callasm, reps)

datasize = (1<<14)

def timeaccess(name, items, mask, store):
    def timereps():
        for i in xrange(reps):
            for j in xrange(items):
               store[j] ^= mask
    timecmd(name, timereps, datasize*reps)

timeaccess("snative", datasize>>2, 3, [0] * (datasize >> 2))
timeaccess("native", datasize>>2, 0xffffffff, [0] * (datasize >> 2))
timeaccess("sarray", datasize>>2, 3, array.array('L', [0] * (datasize >> 2)))
timeaccess("array", datasize>>2, 0xfffffff, array.array('L', [0] * (datasize >> 2)))
timeaccess("sulongs", datasize>>2, 3,  nextarray('L', datasize >> 2))
timeaccess("ulongs", datasize>>2, 0xffffffff,  nextarray('L', datasize >> 2))
timeaccess("ushorts", datasize>>1, 0xffff, nextarray('H', datasize >> 1))

