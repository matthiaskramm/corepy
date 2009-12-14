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

import corepy.arch.spu.isa as spu
import corepy.arch.spu.platform as env
import corepy.spre.spe as spe
#import time


# List of branch/halt instructions that should act as scheduling barriers.
branch_insts = (spu.bi, spu.bihnz, spu.bihz, spu.binz, spu.bisl, spu.bisled,
                spu.biz, spu.br, spu.bra, spu.brasl, spu.brhnz, spu.brhz,
                spu.brnz, spu.brsl, spu.brz, spu.heq, spu.heqi, spu.hgt,
                spu.hgti, spu.hlgt, spu.hlgti)

# List of branch hint instructions for choosing special hint heuristics.
hint_insts = (spu.hbrr, spu.hbra, spu.hbr)

# List of instructions that do not write to their (first) operand.
nowrite_insts = (spu.nop, spu.stqx, spu.stqd, spu.stqa, spu.stqr, spu.wrch,
                 spu.mtspr)

# List of all load/store instructions for hbr heuristic
#ls_insts = (spu.lqa, spu.lqd, spu.lqr, spu.lqx, spu.stqa, spu.stqd, spu.stqr,
#             spu.stqx)


# A basic block.  Built by the DAG generation stage, then used by the topo
# sort.  Blocks can contain at most one label and one branch.  Every time a
# label is encountered by the DAG generation, a new block is created starting
# with that label.  Likewise, every time a branch is encountered, it is added
# to the current block and a new block is started.
class sched_block(object):
  def __init__(self, label = None):
    self.label = label  # Label at start of this basic block
    self.branch = None  # Branch at end of this basic block
    self.inst_cnt = 0   # Number of instructions in the DAG

    self.critpath = {} # Critical path latency for each instruction

    # DAG dictionaries:
    # For each inst, a list of insts that depend on it
    self.g_out = {None:[]}

    # A list of each inst this inst depends on, including None
    self.g_in = {None:[]}

    # Count of actual insts this inst depends on, in the same block
    self.g_incnt = {None:0}

    # Set of instructions with no unresolved dependences in this block
    self.start = []
    return


def heurcompare_block(a, b, pipe, g_maxdist, blocks, block_ind):
  # Use a special heuristic for branch hints to prevent them from being
  # scheduled too far from their branch.
  if isinstance(b[0], hint_insts):
    # Branch hint, figure out how many pending instructions.

    # Extract the label referring to the target branch.
    lbl = b[0]._operand_iter[0]

    # Iterate forward from the current block, adding up the inst_cnt for each
    # block until the block with the target branch is found.
    pending_insts = 0
    for i in xrange(block_ind, len(blocks)):
      pending_insts += blocks[i].inst_cnt
      if blocks[i].branch is not None:
        pending_insts += 1
        if blocks[i].label == lbl:
          break

    # If the hint is too far away from the branch target, don't select it.
    if pending_insts > 250:
      return a

  # Force negative stall counts to -1
  astall = a[1]
  if astall < 0:
    astall = -1
  bstall = b[1]
  if bstall < 0:
    bstall = -1

  # Choose the lowest stall time
  if astall < bstall: # stall check
    return a
  elif astall > bstall:
    return b

  # Match the current pipeline
  #else astall == bstall:
  if a[0].cycles[0] != b[0].cycles[0]: # pipe check
    if a[0].cycles[0] == pipe:
      return a
    elif b[0].cycles[0] == pipe:
      # Would a stall occur if b went one cycle sooner?
      # TODO - results in slower code, less pipelining.. why?
      #if bstall == 0 and b[0].cycles[0] == 1 and pipe == 1:
      #  return a
      return b

  # Choose the longest distance to the end of the stream
  if g_maxdist[b[0]] > g_maxdist[a[0]]:
    return b

  return a


# TODO - maybe do this iteratively from a set of stop nodes?
def critpath_block(blocks):
  g_critpath = {}

  def critpath_rec(g_out, inst):
    mx = 0
    for d in g_out[inst]:
      # Find the critpath for this d
      if d not in g_critpath:
        critpath_rec(g_out, d)

      # Is the distance through this dep longer?
      if g_critpath[d] > mx:
        mx = g_critpath[d]

    g_critpath[inst] = mx + inst.cycles[1]
    return

  for block in reversed(blocks):
    for s in block.start:
      critpath_rec(block.g_out, s)

  return g_critpath


def isched_gen_blocks(scode):
  # deps is a dictionary indexed by registers, mapping to a tuple containing: 
  #  last inst to write this reg
  #  list of insts that read this reg since the last write
  deps = {}

  # Start with a new, empty block
  #block = {'label':None, 'branch':None, 'inst_cnt':0,
  #    'g_out':{None:[]}, 'g_in':{None:[]}, 'g_incnt':{None:0}, 'start':[]}
  block = sched_block()

  blocks = [block]

  # Go through each object in scode:
  for obj in scode:
    if isinstance(obj, spe.Label):
      if block.label is not None or len(block.g_out) > 1:
        # Either the current block already has a label or the DAG is not empty;
        # start a new block

        block = sched_block(label = obj)
        blocks.append(block)

      else:
        # No label is set and the DAG is empty; set label in this new block.
        block.label = obj
    elif isinstance(obj, branch_insts):
      # Set this branch in the current block and start a new block.
      block.branch = obj
      block.inst_cnt += 1

      block = sched_block()
      blocks.append(block)

      # TODO
      # Ideally we want to account for branch register deps when scheduling.
      # Adding the deps themselves is fine, but the only place they would come
      # into effect is for the critpath heuristic.
      # Critpath would need to be updated to calculate for branches also.
      #  Doing so would require tracking the critpath for an entire block,
      #  which probably isn't that hard -- its the max of the critpaths of the
      #  start nodes for each block.

    elif isinstance(obj, spe.Instruction):
      # Add this instruction to the DAG
      block.g_out[obj] = []
      block.g_in[obj] = []
      block.g_incnt[obj] = 0
      block.inst_cnt += 1

      # Go through each operand, adding register deps if any
      for (k, op) in enumerate(obj._operand_iter):
        if not isinstance(op, env.SPURegister):
          continue

        try:
          d = deps[op]
        except KeyError:
          d = deps[op] = (None, [])

        if k == 0 and not isinstance(obj, nowrite_insts):
          # Write dependence -- must go after last write, and all reads since
          #  the last write

          # Compute the stall time - latency of write dependence, or 1
          stall = 1
          if d[0] is not None:
            # Increment incnt only if dep is not None, and in the same block
            if d[0] in block.g_out:
              block.g_incnt[obj] += 1
            stall = d[0].cycles[1]

          # Depend on the last write.
          block.g_in[obj].append((d[0], stall))

          # This is tricky.. iterate back through the blocks looking for the
          # dependence, starting with the current block.
          for b in reversed(blocks):
            if d[0] in b.g_out:
              b.g_out[d[0]].append(obj)
              break

          # Depend on all reads since the last write.
          for r in d[1]:
            block.g_in[obj].append((r, 1))

            # This is tricky.. iterate back through the blocks looking for the
            # dependence, starting with the current block.
            for b in reversed(blocks):
              if r in b.g_out:
                b.g_out[r].append(obj)

                # Increment the incnt only if this dep is in the same block
                if b == block:
                  block.g_incnt[obj] += 1
                break
  
          # Set this inst as the last to write this operand,
          #  and clear the list of reads since the last write.
          deps[op] = (obj, [])
        elif obj != d[0] and obj not in d[1]:
          # Careful to not add the same dep multiple times
          # Read dependence -- must go after last write
          stall = 0
          if d[0] != None:
            # Increment incnt only if dep is not None, and in the same block
            if d[0] in block.g_out:
              block.g_incnt[obj] += 1
            stall = d[0].cycles[1]

          # Add the last write to the list of insts this inst depends on
          block.g_in[obj].append((d[0], stall))

          # Add this inst to the list of insts that depend on the last write
          # This is tricky.. iterate back through the blocks looking for the
          # dependence, starting with the current block.
          for b in reversed(blocks):
            if d[0] in b.g_out:
              b.g_out[d[0]].append(obj)
              break

          deps[op][1].append(obj)
     
      if block.g_incnt[obj] == 0:
        block.start.append(obj)

    else:
      raise TypeError("Unexpected object type in stream:" + str(type(obj)))
  # end for obj in scode

  return blocks;


def isched(scode):
  old_active_code = spu.get_active_code()
  spu.set_active_code(None)

  # Generate the instruction dependence DAG(s)
  blocks = isched_gen_blocks(scode)

  # For each instruction, compute the max cycles to the end of the code
  g_critpath = critpath_block(blocks)

  # Apply heuristics to build an optimized InstructionStream
  fcode = scode.prgm.get_stream()

  inst_cycle = {} # For each inst, the cycle number it has in the code

  lastpos = -1    # Index of last instruction in the stream (excludes labels!)
  pipe = 0        # Current pipeline (0 = even, 1 = odd)
  cycle = 0       # Current cycle number

  for (ind, block) in enumerate(blocks):
    if block.label is not None:
      fcode.add(block.label)

    start = block.start
    g_in = block.g_in
    g_incnt = block.g_incnt
    g_out = block.g_out

    while len(start) > 0:
      # Apply heuristics to find the best instruction in the queue.
    
      # For each inst in start, compute the minimum stall time
      # TODO - cache this instead of computing each time?
      #  Do this by computing the stall time when an inst is added to start.
      #  Each time the cycle number is moved forward, reduce the stall time
      #  by that number of cycles for each inst in start.
      # TODO - idea from I think Muchnick -- keep a start Q of no-stall nodes,
      #  and a Q of nodes that would stall.  Then just pull from no-stall nodes
      #  unless empty, in which case fall back to the stall Q
      #   would make it easy(er) to do cached stall counts

      best = (None, 999)
      for s in start:
        # Find the stall time of s, or maximum delay for all its deps
        maxstall = 0
        for d in g_in[s]:
          if d[0] == None:
            continue

          # Compute stall time for this dep
          stall = d[1] - (cycle - inst_cycle[d[0]])
          if stall > maxstall:
            maxstall = stall

        best = heurcompare_block(best, (s, maxstall), pipe, g_critpath, blocks, ind)

      inst = best[0]

      start.remove(inst)
      cycle += best[1] + 1

      block.inst_cnt -= 1
      fcode.add(inst)

      # Dual issue? if so, adjust the cycle back one.
      # Careful, lastpos starts out as -1.  However the pipe also starts out
      # as 0, so the first part of the conditional will fail before lastpos
      # is used.

      # Ah - if a label occurs first in the stream, followed by say an ai,
      # this will fail
      previnst = fcode[lastpos]
      if (pipe == inst.cycles[0] == 1 and
          previnst.cycles[0] == 0 and 
          inst_cycle[previnst] == cycle - 1):
        cycle -= 1

      inst_cycle[inst] = cycle

      lastpos = len(fcode) - 1
      pipe = (pipe + 1) & 1

      # Evaluate all the instructions that depend on this inst.
      # Can any be added to start?
      for d in g_out[inst]:
        # Skip this d if it's not in the current block
        if d not in g_incnt:
          continue

        g_incnt[d] -= 1
        if g_incnt[d] == 0:
          start.append(d)

        # Does d have depend on any insts in start?
        # If so, move those to the front of start
        # Why does this still matter?  It affects ties in the heuristic..
        else:
          # d depends on inst, and at least 1 other inst not in start.
          # look at the insts d depends on.  if any are in start, move
          # them to the front of the start set.
          # how does this help?  helps insts closer to getting into start,
          #  get in sooner.  get a larger start set for choosing best inst
          for e in g_in[d]:
            if e[0] in start:
              start.remove(e[0])
              start.insert(0, e[0])

    # end while len(start) > 0
    if block.branch is not None:
      fcode.add(block.branch)
  # end for block in blocks

  spu.set_active_code(old_active_code)
  return fcode



if __name__ == '__main__':
  import corepy.lib.printer as printer
  import corepy.arch.spu.lib.util as util

  prgm = env.Program()
  code = prgm.get_stream()
  spu.set_active_code(code)

  r_cnt = prgm.acquire_register()
  r_cmp = prgm.acquire_register()
  r_sum = prgm.acquire_register()

  spu.il(r_cnt, 32)
  spu.il(r_sum, 0)
  lbl_loop = prgm.get_unique_label("LOOP")
  code.add(lbl_loop)

  spu.ai(r_sum, r_sum, 1)

  spu.ceqi(r_cmp, r_cnt, 2)
  spu.brz(r_cmp, lbl_loop)

  spu.ai(r_sum, r_sum, 10)

  #src = prgm.acquire_register()
  #tmp = prgm.acquire_registers(3)
  #dst = prgm.acquire_registers(2)

  #spu.il(tmp[0], 1)
  #spu.il(tmp[1], 2)
  #spu.il(tmp[2], 3)
  #spu.fma(src, tmp[0], tmp[1], tmp[2])
  #spu.fa(dst[0], src, src)

  #spu.fnms(src, tmp[0], tmp[1], tmp[2])
  #spu.fs(dst[1], src, src)

  fcode = isched(code)

  code.print_code()
  fcode.print_code()

  #printer.PrintInstructionStream(code, printer.SPU_Asm(comment_chan = True))
  #print "# %d instructions" % len(code)
  #printer.PrintInstructionStream(fcode, printer.SPU_Asm(comment_chan = True))

