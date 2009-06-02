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

#------------------------------------------------------------------------------

# AlignedMemory contributed by Mark Jeffree
# This is to manage aligned memory allocation on behalf of an SPU, but it would
# also work for the main PPC memory.

#------------------------------------------------------------------------------
#
# A memory pool is created by instantiation of the Allocator. eg
#
#    myPool = Allocator(base, size)
#
# If the base and size parameters are not aligned, they are padded inwards,
# ie the base is increased and the size is decreased.
#
# A request is made from that pool as follows.
#
#    myHandle = myPool.alloc(size)
#
# Whether the request was for a large or small block, a MemoryHandle is
# returned that has "addr" and "size" fields and a "free()" method.
#
# The space must be returned to the pool explicitly:
#
#    myHandle.free()
#
# All address and sizes are in bytes.
#
#------------------------------------------------------------------------------

# We have two important alignment criteria.
# We need to align memory objects on 16 byte boundaries to ensure correct
# quadword access. We would like to align memory objects on 128 byte boundaries,
# for efficient DMA. If a memory object is smaller than 128 bytes
# (ie 8 quadwords), there will always be the same DMA cost in transfer, so we
# can share the smaller spaces without penalty.

# All memory access will be made in quadwords, and smaller sections handled
# by register manipulation. (It is especially important to note that small 
# (<16 byte) writes must be managed as read modify write cycles.

# The approach taken here uses two mechanisms. Any request larger (or equal to)
# than 128 bytes is rounded up to the next 128 bytes and allocated on a 128
# byte boundary. The blocks are managed as as double linked list that contain
# descriptors for blocks that are both in-use or free. This allows the list to 
# remain in order, which makes coalescence more efficient.

# Any request smaller than 128 bytes is rounded up to the next
# 16 bytes and allocated out of an existing shared block that in multiples of
# 16 bytes. If a suitable shared block cannot be found, a new request for an
# empty 128 byte shared block is made from the pool.

# When the entire 128 byte shared block is freed af all small requests, it is
# returned the large block free pool.

# This serves a couple of purposes. It reduces the number of small blocks
# cluttering up the dynamic memory block list, which makes the list faster
# to traverse, and it tends to cluster the little requests together, which
# reduces fragmentation.

# Each shared block needs to be managed, but this is a relatively small
# problem, because there are only a small number of slots, and we are
# not too concerned about wasting a little space. We can apply some
# simplifications.

# Small requests will never span two shared blocks.
# We never try to reclaim space from shared blocks. When all the tenants
# have vacated, we release the whole block back to the pool.

# This means that we can test whether a shared block is big enough to meet 
# the request with a single compare, and we never concern ourselves with the
# small scale coalescence.

BLOCK_ALIGNMENT = 128
TENANT_ALIGNMENT = 16
SHARED_BLOCK_SIZE = 128

BLOCK_MASK = -BLOCK_ALIGNMENT
TENANT_MASK = -TENANT_ALIGNMENT

class MemoryHandle(object):
    
    # This is the handle provided to the user to obtain
    # address and size, as well as free the memory, for 
    # memory in a shared block.
    
    # It does not form an element of any list
    
    def __init__(self, addr, size, block):
        self.addr = addr
        self.size = size        
        self._block = block
        
    def free(self):        
        self._block.free()
        
                             
class Allocator(object):

    class BlockDescriptor(object):
        
        # This is the internal descriptor is for a block of memory
        # that may or may not be free and may or may not be shared.
        # The aligned memory pool contains a double linked list of
        # these descriptors.
        
        def __init__(self, addr, size):
            
            self.addr = addr
            
            # if the block is shared, the size indicates size remaining
            
            self.size = size
            self.prev = None
            self.next = None
            self.isFree = False
            self.isShared = False
            
            # the following is only relevent if the block is shared
    
            self.numberOfTenants = 0
            
        def release(self):
            # mark as unused, in case we can't merge    
            self.isFree = True
            
            # try to merge to the left and then right
        
            if self.prev.isFree:
                self.prev.next = self.next
                self.prev.size += self.size
                self.next.prev = self.prev
                self = self.prev  # prepare for the RHS merge
                # disappear and let Python reap the old block reference
        
            if self.next.isFree:
                self.size += self.next.size
                self.next = self.next.next
                self.next.prev = self
                # and the old block.next is reaped by Python garbage collection
    
        def free(self):
            # If this is a shared block, decrement the tenant count and
            # if that completely clears the block, release it.
            # Note that we do NOT try to reclaim the tenant space
            # until all the tenants have gone. There is a memory
            # overhead to be paid for this decision but it is relatively
            # small and saves management overhead for small gains
            
            # If it isn't shared, just release the block
            
            if self.isShared:    
                self.numberOfTenants -= 1
                if self.numberOfTenants < 1:
                    # we need to "unshare" and resize, to be ready
                    # for possible coalescence
                    self.isShared = False
                    self.size = SHARED_BLOCK_SIZE
                else:
                    return
    
            self.release()

    def __init__(self, base, size):

        # Align the memory pool
        requestBase = (base+BLOCK_ALIGNMENT-1) & BLOCK_MASK        
        requestSize = (size - (requestBase - base) ) & BLOCK_MASK
        
        # create a three block double linked list.
        # the two empty end sentinals remove the need to test for
        # special cases in the alloc and free functions. The base address
        # and the size of the start and end sentinals indicate what
        # padding we needed to align the memory pool
        
        # create a dummy start sentinal that can never service a request       
        self.firstBlock = Allocator.BlockDescriptor(
            base, requestBase - base)
    
        # hand graft an end sentinal as the following block
        self.firstBlock.next = Allocator.BlockDescriptor(
            requestBase+requestSize, (base+size) - (requestBase+requestSize))
        self.firstBlock.next.prev = self.firstBlock

        # hand insert the actual memory between the sentinals
        freeMemory = Allocator.BlockDescriptor(requestBase, requestSize)
        freeMemory.isFree = True
        freeMemory.next = self.firstBlock.next
        freeMemory.prev = self.firstBlock
      
        self.firstBlock.next = freeMemory
        freeMemory.next.prev = freeMemory
        
        # the shared blocks are a single linked list
        self.sharedBlock = None

    def _requestBlock(self, requestSize):

        # This is a firstfit search that splits the allocation from the
        # *start* of the the first block that meets the request.
        
        assert requestSize == (requestSize+BLOCK_ALIGNMENT-1) & BLOCK_MASK
        
        block = self.firstBlock.next # we know the sentinal can be skipped
        
        while block <> None:
            if block.isFree:
                if requestSize == block.size:
                    # perfect fit
                    block.isFree = False
                    
                    return block
                
                elif requestSize < block.size:
                    # we make a new block to fit
                    newBlock = Allocator.BlockDescriptor(
                        block.addr, requestSize)
                    block.addr += requestSize
                    block.size -= requestSize
    
                    # since we are here, list insertion is easy
                    newBlock.prev = block.prev
                    block.prev = newBlock
                    newBlock.next = block
    
                    # our start sentinal makes the following easy and safe
                    newBlock.prev.next = newBlock
                    
                    return newBlock
                
            block = block.next
       
        raise Exception, "Allocate Failed - pool is all in use"

    def alloc(self, size):
        
        # If the request is big enough for a non-shared block
        # get it directly
        
        if size >= BLOCK_ALIGNMENT:
            requestSize = (size+BLOCK_ALIGNMENT-1) & BLOCK_MASK
            block = self._requestBlock(requestSize)
        
            return MemoryHandle(block.addr, requestSize, block)
        
        # Otherwise, look for a share
        
        requestSize = (size+TENANT_ALIGNMENT-1) & TENANT_MASK
        
        # Traverse the block list to find the first shared
        # block that has enough space to meet the (expanded) request
        block = self.firstBlock.next # we know the first entry can be skipped
        
        while block <> None:
            if block.isShared:
                if block.size >= requestSize:
                    break
            block = block.next
            
        # There wasn't a shared block with enough space, allocate a new
        # shared block from the general pool.
        
        if block == None:
            # Ask a grown up to get you a new block
            block = self._requestBlock(SHARED_BLOCK_SIZE)
            block.isShared = True
        
        # Either way, now we have a block that will do the trick
        
        # The small allocation is taken from the *end* of the shared
        # block, so that the shared block address doesn't change and
        # so that the calculation of small base address is easier.
        
        block.numberOfTenants += 1
        block.size -= requestSize
        baseAddress = block.addr + block.size
                
        return MemoryHandle(baseAddress, requestSize, block)

    def _printMemList(self):
    
        block = self.firstBlock
        
        print "MemPool:"
        while block <> None:
            print "addr: ", hex(block.addr), "  size: ", \
                  hex(block.size), "  free: ", block.isFree, \
                  "  shared: ", block.isShared
            block = block.next
        print



if __name__=='__main__':

    # Create the pool
    
    spuMem = Allocator(10,4026)   
    
    spuMem._printMemList()
    
    # some testing
    
    a = spuMem.alloc(200)
    spuMem._printMemList()    
    
    b = spuMem.alloc(390)
    spuMem._printMemList()
    
    a.free()
    spuMem._printMemList()

    c = spuMem.alloc(200)
    spuMem._printMemList()
    
    b.free()
    spuMem._printMemList()
    c.free()
    spuMem._printMemList()
    
    d = spuMem.alloc(60)
    print "d.addr = ", d.addr
    
    spuMem._printMemList()

    e = spuMem.alloc(30)
    print "e.addr = ",e.addr
    spuMem._printMemList()

    d.free()
    spuMem._printMemList()
    
    f = spuMem.alloc(60)
    print "f.addr = ",f.addr
    spuMem._printMemList()

    e.free()
    spuMem._printMemList()

    f.free()
    spuMem._printMemList()
    
