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


# Idea -- interleaving still requires that the programmer use different sets of
#  register for each block to be interleaved.  We could detect the 'regions' in
# which say a temporary register is used, and change the registers to allow
# interleaving two regions.
# - Maybe have the programmer acquire/release temp regs around each region.
#   Then this acquire/release timing could be included in the stream for the
#   scheduler to use.  Still, the same registers are used, meaning some regions
#   would have to be changed to allow the interleaving.
# - Abstract registers a little better, and delay register number assignment.
#   Then different 'virtual' registers are used in each region.  Actual regs
#   are assigned later.. as part of the scheduling process?
#   - This could make the scheduler pretty complicated.. would want to add
#     heuristic criteria to minimize register usage when other criteria is met.
#     i.e. If two regions interleaved are enough to avoid all stalls, then
#     don't interleave four regions, itll just waste registers.

# Feature TODO:
# - Allow a branch hint to be specified and scheduled in as appropriate
#   Needs to be >8, < 125 instructions from the end of the stream
# - Do proper scheduling of DMA commands:
#   - don't move completion before the command is issued
#   - issue wrch commands according to SPU constraints
#     - what are they exactly? the cmd i think has to go last
# - Consider using collections.deque for lists



# Use the scheduling barriers to organize the stream into basic blocks.
# Buid a list of DAGs (each DAG is a basic block), then schedule each one.
# If a label is followed immediately by a branch, make them be one block.
# Keep adding deps even across blocks; but schedule the insts in one block
# at a time.  This should allow latencies to be dealt with across block
# boundaries.

# How do I know when to start a new block?
# - Maybe labels should appear at the top of their block,
#   and branches at the end of their block.
# - with each block, store the start/end barrier separately from the
#   DAG.  This way it is easy to schedule these first/last.
#   A hinted branch is a block with no DAG, just label and branch.
#   Blocks may have no barriers; this is OK
# Which block does the scheduling barrier go to?

# TODO - how should branch inst reg deps be handled?
#  branches can't be moved, but the insts that satisfy the deps can
# How should the max_depth calculation deal with barriers?
#  One way is to ignore them; could lead to suboptimal results
#  labels have zero latency and zero stall time.. ignore them safely?
#  branches have latency and potential stall time.
# Need to make the heuristic aware of blocks some how --
#  pass it the list of blocks in the first place!
#   can work backwards through the list of blocks; would reduce recursion and give speedup!
#  calculate the latency of an entire block
#   latency of an entire block is the longest dependency chain cycle count
#    so, while computing all the maxdepths for the insts in a block,
#    keep track of the max value; the max is the latency for the block.
#    a branch has a latency of the following block + 1
#     instructions don't depend directly on branches; the branch latency is irrelevant


# A block contains:
#  start    Starting label (possibly None)
#  end      Ending branch  (possible None)
#  g_out    For each inst, a list of insts that depend on it
#  g_in     A list of each inst this inst depends on, including None
#  g_incnt  count of actual insts this inst depends on
#  start    A list of instructions with no (unresolved) dependences 

# So, the DAG building algorithm:
# - register deps are global, for cross-block register stall handling
# Start a new block
# while more code to process:
#   is next object a label? 
#     if so, does the current block already have a label start?
#       if so, start a new block, set this object as the label start, continue
#       if not, set this object as teh label start, continue
#         CAREFUL -- only want to set the start if the DAG is empty
#   is next object a branch?
#     if so, set thisi object as the branch end, start a new block, continue
#   else, add object to DAG as normal

# The topo sort algorithm:
#  for each block:
#   if block has a start label, add it
#   process DAG as per normal
#   if block has a end branch, add it


# Why am i doing this?
# - adding tons of deps in the DAG seems a little silly
# - basic blocks sound more intelligent
# - performance; adding all the edges seems to slow things down





# ideas for branch hinting:
#  have a function that takes two labels -- one is the branch's target, the
#  other precedes the branch itself.
# Find the branch, then search backwards 11 cycles and insert a nop/hbrr pair.
#  maybe use the hbr a-form and give a literal address?
#  if a label or conditional branch is reached before 11 cycles, print a
#   warning (return a warning?) and insert the hint as best as possible.

# if the scheduler encounters a hint, note the surrounding scheduling barriers
#  and leave it out of the scheduling.  At the end, come back and insert a
#  nop/hint pair at some point within the scheduling barrier region,
#  moving it to fit in the 11 cycle to 255 instructino range if possible.
# Maybe only do this if we know where the targets of the hint are --
#  if we don't, we cant intelligently place it, so treatit like a noop

# an hbr instruction has a specific latency.. can use that for the minimum
# distance from the branch. but again, what about max distance?

# max distance heuristic:
#  consider the label the hint references, for the branch itself
#  does this label depend on the branch hint?
#   if so, distance = incnt of this label
#   if not, recurse: dist = incnt of the label + incnt of its prev barrier
#     does prev barrier depend on hint? if so, we're done
#      this must occur at some point before prev barrier is none
#       although, checking prev_barrier == none might speed things up
# this won't work in the case that the hbr is AFTER the branch it is hinting.
#  when this happens, the recursive max distance WILL hit the none barrier
#   when the none barrier is hit, just treat the hbr as a noop.

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


def isched_block(scode):
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










def heurcompare(a, b, pipe, g_maxdist, g_in, g_incnt):
  # hint heuristics:
  #  if pending_insts > 250 (?), do not select the hint!
  # Use a special heuristic for branch hints to prevent them from being
  # scheduled too far from their branch.
  if type(b[0]) in hint_insts:
    # Branch hint, figure out how many pending instructions.
    lbl = b[0]._operand_iter[0]
    pending_insts = g_incnt[lbl]

    # Never choose a hint if it's too far from its hinted branch.
    if pending_insts > 200:
      return a

    # Repeatedly go back through the scheduling barriers, adding to
    # pending_insts until the barrier that depends directly on this hint is
    # found.
    while b[0] not in [d[0] for d in g_in[lbl]]:
      # The DAG construction always places the last barrier dep first, so
      # just check that one element.
      if not (isinstance(g_in[lbl][0], spe.Label) or
          type(g_in[lbl][0]) in branch_insts):
        # Reached the beginning of the code.  This means the hbr is after the
        # branch it hints, so just treat it like an lnop.
        break

      lbl = g_in[lbl][0]
      pending_insts += g_incnt[lbl]

      # Never choose a hint if it's too far from its hinted branch.
      if pending_insts > 20:
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

  # Avoid choosing a load/store if too many have been issued
  #if a in ls_insts and LS > 15:
  #  return b
  return a


# TODO - maybe do this iteratively from a set of stop nodes?
def maxdist(start, g_in, g_out):
  g_maxdist = {}

  def maxdist_rec(inst):
    mx = 0
    for d in g_out[inst]:
      # Find the maxdist for this d
      if d not in g_maxdist:
        maxdist_rec(d)

      # Is the distance through this dep longer?
      if g_maxdist[d] > mx:
        mx = g_maxdist[d]

    if isinstance(inst, spe.Instruction):
      g_maxdist[inst] = mx + inst.cycles[1]
    elif isinstance(inst, spe.Label):
      # Labels have no latency
      g_maxdist[inst] = mx
    return

  for s in start:
    maxdist_rec(s)

  return g_maxdist


def isched_generate_dag(code, g_out, g_in, g_incnt, start):
  # deps is a dictionary indexed by registers, mapping to a tuple containing: 
  #  last inst to write this reg
  #  list of insts that read this reg since the last write
  deps = {}
  #g_out = {}    # For each inst, a list of insts that depend on it
  #g_in = {}     # A list of each inst this inst depends on, including None
  #g_incnt = {}  # count of actual insts this inst depends on
  #start = []

  g_out[None] = []
  g_in[None] = []
  g_incnt[None] = 0

  barrier = None # Reference to most recent scheduling barrier, ie label or branch
  barrier_deps = [] # Instructions added to DAG since most recent barrier

  # First just build a graph using operands to find dependences
  for inst in code:
    if isinstance(inst, spe.Label):
      # Labels depend on every inst since the last barrier, plus the barrier.
      #  For scheduling, labels have a latency of 0
      g_out[inst] = []
      g_in[inst] = [(barrier, 0)]

      g_incnt[inst] = len(barrier_deps)
      if barrier is not None:
        g_incnt[inst] += 1

      g_out[barrier].append(inst)

      for dep in barrier_deps:
        # This label depends on every instruction since the last barrier.

        # If this dep is a hint instruction, check if this label is the branch operand.
        # If it is, set the stall time properly.
        stall = 0
        if type(dep) in hint_insts and dep._operand_iter[0] == inst:
          stall = dep.cycles[1]

        g_out[dep].append(inst)
        g_in[inst].append((dep, stall))

      # Reset the barrier and its dependences
      barrier = inst
      barrier_deps = []

      # Very unlikely, but a branch could have no deps.
      if g_incnt[inst] == 0:
        start.append(inst)
 
    elif type(inst) in branch_insts:
      # Branches depend on every inst since the last barrier, plus the barrier.
      #  Also, the branch may have 1-2 read-after-write dependences, which must
      #  be handled specifically to determine the proper latency.
      # Add the branch to the graph
      g_out[inst] = []

      # Depend on the last barrier.
      if barrier is not None:
        g_in[inst] = [(barrier, 0)]
        g_incnt[inst] = 1
        g_out[barrier].append(inst)
      else:
        g_in[inst] = []
        g_incnt[inst] = 0
 

      # First, handle any register dependences.
      # The barrier is always a label or branch, meaning it never writes
      # to a register.  Thus this branch instruction can't have a register
      # dependence on the last barrier.  This allows us to just always
      # add the last barrier as a dep before processing register dependences.
      reg_deps = []
      for op in inst._operand_iter:
        if isinstance(op, env.SPURegister):
          try:
            d = deps[op]
          except KeyError:
            d = deps[op] = (None, [])

          # Only a read dependence can occur here.
          if inst != d[0] and inst not in d[1]:
            # Careful to not add the same dep multiple times
            # Read dependence -- must go after last write
            stall = 0
            if d[0] != None:
              stall = d[0].cycles[1]
              g_incnt[inst] += 1

              # Add to the list of register dependences
              reg_deps.append(d[0])

            # Add the last write to the list of insts this inst depends on
            g_in[inst].append((d[0], stall))

            # Add this inst to the list of insts that depend on the last write
            g_out[d[0]].append(inst)
            deps[op][1].append(inst)

      # Depend on all instructions since the last barrier.
      for dep in barrier_deps:
        # This branch depends on every instruction since the last barrier.
        # Register dependences are already taken care of, skip those.
        if dep not in reg_deps:
          g_in[inst].append((dep, 0))
          g_incnt[inst] += 1
          g_out[dep].append(inst)

      # Reset the barrier and its dependences
      barrier = inst
      barrier_deps = []

      # Very unlikely, but a branch could have no deps.
      if g_incnt[inst] == 0:
        start.append(inst)
 
    elif isinstance(inst, spu.Instruction):
      # Add this instruction to the graph
      g_out[inst] = []
      g_in[inst] = []
      g_incnt[inst] = 0

      # Add a dependence on the last barrier, if not None
      barrier_deps.append(inst)
      if barrier is not None:
        g_out[barrier].append(inst)
        g_in[inst].append((barrier, 0))
        g_incnt[inst] += 1

      # Go through each operand, adding dependencies if any
      for (k, op) in enumerate(inst._operand_iter):
        # If this is a reg, add a dependency.  If the reg was never written by
        # a prior instruction, depend on the None instruction, i.e. depend on 
        # something before the code starts.
        # TODO - for now assume the first operand is never read AND written
        #  What instructions do this?
        if isinstance(op, env.SPURegister):
          try:
            d = deps[op]
          except KeyError:
            d = deps[op] = (None, [])

          if k == 0 and not isinstance(inst, nowrite_insts):
            # Write dependence -- must go after last write, and all reads since
            #  the last write

            # Compute the stall time - latency of write dependence, or 1
            stall = 1
            if d[0] is not None:
              g_incnt[inst] += 1
              stall = d[0].cycles[1]

            # Depend on the last write, and all reads since.
            g_in[inst].append((d[0], stall))
            g_out[d[0]].append(inst)

            for r in d[1]:
              g_in[inst].append((r, 1))
              g_out[r].append(inst)
            g_incnt[inst] += len(d[1])

            # Set this inst as the last to write this operand,
            #  and clear the list of reads since the last write.
            deps[op] = (inst, [])
          elif inst != d[0] and inst not in d[1]:
            # Careful to not add the same dep multiple times
            # Read dependence -- must go after last write
            stall = 0
            if d[0] != None:
              stall = d[0].cycles[1]
              g_incnt[inst] += 1

            # Add the last write to the list of insts this inst depends on
            g_in[inst].append((d[0], stall))

            # Add this inst to the list of insts that depend on the last write
            g_out[d[0]].append(inst)
            deps[op][1].append(inst)
     
      if g_incnt[inst] == 0:
        start.append(inst)

#  for k in g_in.keys():
#   print k, "INCNT", g_incnt[k]
#   for d in g_in[k]:
#     print "    ", d[0], "DELAY", d[1]
#  print "start", start

  return (g_out, g_in, g_incnt, start)


def isched(scode):
  old_active_code = spu.get_active_code()
  spu.set_active_code(None)

  g_out = {}    # For each inst, a list of insts that depend on it
  g_in = {}     # A list of each inst this inst depends on, including None
  g_incnt = {}  # Count of actual insts each inst depends on
  start = []

  # Generate the instruction dependence DAG
  isched_generate_dag(scode, g_out, g_in, g_incnt, start)

  # For each instruction, compute the max cycles to the end of the code
  g_maxdist = maxdist(start, g_in, g_out)

  # Apply heuristics to build an optimized InstructionStream
  fcode = scode.prgm.get_stream()

  inst_cycle = {}   # For each inst, the cycle number it has in the code

  lastpos = -1   # Index of last instruction in the stream (excludes labels!)
  pipe = 0
  cycle = 0 # Current cycle number
  LS = 0    # Count of loads/stores issued in a row

  while len(start) > 0:
    # Labels and branches are scheduling barriers.  As such, they will only
    # appear in the start queue alone, i.e. they are the only object in the
    # queue.  As such, they are handled specially; no heuristics are needed as
    # no choice needs to be made.
    #print "start", start
    if len(start) == 1 and (isinstance(start[0], spe.Label) or start[0] in branch_insts):
      inst = start[0]
      fcode.add(inst)
      start.remove(inst)
      inst_cycle[inst] = cycle

      if not isinstance(inst, spe.Label):
        lastpos = len(fcode) - 1
        pipe = (pipe + 1) & 1

    else:
      # Normal case -- all instructions excluding branches.
      # Apply heuristics to find the best instruction in the queue.
    
      # For each inst in start, compute the minimum stall time
      # TODO - cache this instead of computing each time?
      #  Do this by computing the stall time when an inst is added to start.
      #  Each time the cycle number is moved forward, reduce the stall time
      #  by that number of cycles for each inst in start.

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

        best = heurcompare(best, (s, maxstall), pipe, g_maxdist, g_in, g_incnt)

      inst = best[0]

      # Increment LS counter for loads and stores.
      # Reset it for any stalls, or non-LS insts issued on an odd address
#      if isinstance(inst, ls_insts):
#        LS += 1
#      elif best[1] > 0 or pipe == 1:
#        LS = 0

#      did_hbr = False
#      if LS > 14 and pipe == 1 and isinstance(inst, ls_insts):
#        # Issue hbrp instead to prevent instruction prefetch starvation
#        LS = 0
#        #inst = spu.hbr(0, fcode.r_zero, True)
#        inst = spu.lnop()
#        cycle += 1
#        did_hbr = True
#        print "did hbr"
#      else:
      start.remove(inst)
      cycle += best[1] + 1

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

    # An inserted hbr won't be in the DAG, so skip adding nodes to start
#     if did_hbr:
#       continue

    # Evaluate all the instructions that depend on this inst.
    # Can any be added to start?
    for d in g_out[inst]:
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

  #for k in g_incnt.keys():
  #  if g_incnt[k] > 0:
  #    print "# ERROR, this inst still has deps:", k

  #print "# cycles", cycle
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

  fcode = isched_block(code)

  code.print_code()
  fcode.print_code()

  #printer.PrintInstructionStream(code, printer.SPU_Asm(comment_chan = True))
  #print "# %d instructions" % len(code)
  #printer.PrintInstructionStream(fcode, printer.SPU_Asm(comment_chan = True))

