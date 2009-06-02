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

import alloc


_DEBUG = False

class extarray(object):
  _typesizes = {'c':alloc.size_char,  'b':alloc.size_char, 'B':alloc.size_char,
      'h':alloc.size_short, 'H':alloc.size_short,
      'i':alloc.size_int, 'I':alloc.size_int,
      'l':alloc.size_long, 'L':alloc.size_long,
      'f':alloc.size_float, 'd':alloc.size_double}

  def __init__(self, typecode, init = None, huge = False):
    self._huge = huge
    self._alloc_len = 0
    self._memory = 0
    self._lock = False

    if huge == True and alloc.has_huge_pages() == 0:
      raise MemoryError("No huge pages available, try regular pages")

    self._set_type_fns(typecode)

    if huge == True:
      self._alloc = alloc.alloc_hugemem
      self._realloc = alloc.realloc_hugemem
      self._free = alloc.free_hugemem
      self._page_size = alloc.get_hugepage_size()
    else:
      self._alloc = alloc.alloc_mem
      self._realloc = alloc.realloc_mem
      self._free = alloc.free_mem
      self._page_size = alloc.get_page_size()


    if init == None:
      self._data_len = 0
    elif isinstance(init, (int, long)):
      self._data_len = init 
      self.alloc(init)
    else:
      self._data_len = len(init) # number of data items
      self.alloc(self._data_len)

      self._memory = self._alloc(self._alloc_len)

      # TODO - something faster?
      for i, j in zip(init, range(0, self._data_len)):
        self._setitem(self._memory, j, i)

    #if _DEBUG:
    #  print "addr: %08x items: %d bytes: %d" % (self._memory,
    #          self._data_len, self._alloc_len)
    return


  def __del__(self):
    if self._memory != None and self._memory != 0 and self._lock != True:
      if _DEBUG:
        print "Freeing memory at 0x%x" % (self._memory)

      try:
        self._free(self._memory)
      except AttributeError:
        print "WARNING could not free memory at 0x%x" % (self._memory)
    return


  def __len__(self):
    return self._data_len


  def _set_type_fns(self, typecode):
    if typecode == 'u':
      raise NotImplementedError("Unicode not supported by extarray")

    try:
      self.itemsize = self._typesizes[typecode]
    except KeyError:
      raise TypeError("Unknown array type '%s' specified" % typecode)

    self.typecode = typecode

    if self.itemsize == 1:
      self._byteswap = lambda a, b: None
    elif self.itemsize == 2:
      self._byteswap = alloc.byteswap_2
    elif self.itemsize == 4:
      self._byteswap = alloc.byteswap_4
    elif self.itemsize == 8:
      self._byteswap = alloc.byteswap_8

    if typecode == 'c' or typecode == 'b':
      self._setitem = alloc.setitem_schar
      self._getitem = alloc.getitem_schar
    if typecode == 'B':
      self._setitem = alloc.setitem_uchar
      self._getitem = alloc.getitem_uchar
    elif typecode == 'h':
      self._setitem = alloc.setitem_sshort
      self._getitem = alloc.getitem_sshort
    elif typecode == 'H':
      self._setitem = alloc.setitem_ushort
      self._getitem = alloc.getitem_ushort
    elif typecode == 'i':
      self._setitem = alloc.setitem_sint
      self._getitem = alloc.getitem_sint
    elif typecode == 'I':
      self._setitem = alloc.setitem_uint
      self._getitem = alloc.getitem_uint
    elif typecode == 'l':
      self._setitem = alloc.setitem_slong
      self._getitem = alloc.getitem_slong
    elif typecode == 'L':
      self._setitem = alloc.setitem_ulong
      self._getitem = alloc.getitem_ulong
    elif typecode == 'f':
      self._setitem = alloc.setitem_float
      self._getitem = alloc.getitem_float
    elif typecode == 'd':
      self._setitem = alloc.setitem_double
      self._getitem = alloc.getitem_double
    return


  def alloc(self, length):
    """Allocate memory for at least length items, without initializing that
       memory.  If the new size is longer then the size required by the items
       stored in the array, every item in the array is preserved.  If the new
       size is shorter, the first length items are preserved."""
    if self._lock == True:
      raise MemoryError("Attempt to allocate with memory lock enabled")

    # Round size to a page
    size = length * self.itemsize
    m = size % self._page_size
    if m != 0:
      size += self._page_size - m

    if self._alloc_len < size:
      self._memory = self._realloc(self._memory, self._alloc_len, size)
      self._alloc_len = size
      if length < self._data_len:
        self._data_len = length
      if _DEBUG:
        print "Allocated %d bytes at 0x%x" % (size, self._memory)
    return


  def append(self, val):
    # TODO - undo the allocation if an exception occurs?
    self.alloc(self._data_len + 1)
    self._setitem(self._memory, self._data_len, val)
    #self.__setitem__(self._data_len, val)
    self._data_len += 1
    return

  def buffer_info(self):
    return (long(self._memory), self._data_len)

  def byteswap(self):
    self._byteswap(self._memory, self._data_len)
    return

  def change_type(self, typecode):
    """Change the type of elements in the array without changing the array data.
       This has an effect similar to typecasting in C.
       The array length (in bytes) must be a multiple of the new type.
       Otherwise, a TypeError exception is raised."""

    if (self._data_len * self.itemsize) % self._typesizes[typecode] != 0:
      raise TypeError("Array length is not a multiple of type '%s'" % typecode)

    self._set_type_fns(typecode)
    return


  def clear(self):
    """Quickly clear all elements to zero."""
    alloc.zero_mem(self._memory, self._alloc_len)
    return


  def copy_direct(self, str):
    """Copy data in str directly into the array, regardless of item size.
       Overwrites any data that may already be in the array, reallocating more
       memory only if needed. 

       Useful for quickly initializing an array with raw data."""

    l = len(str)
    self._data_len = l / self.itemsize
    self.alloc(self._data_len)
    alloc.copy_direct(self._memory, str, l)
    return


  def count(self, val):
    c = 0
    for i in range(0, self._data_len):
      if self.__getitem__(i) == val:
        c += 1
    return c


  def extend(self, iter):
    data_len = self._data_len
    alloc_len = self._alloc_len
    try:
      for i in iter:
        self._setitem(self._memory, self._data_len, i)
        #self.__setitem__(self._data_len, i)
        self._data_len += 1
    except TypeError:
      if alloc_len < self._alloc_len:
        self._memory = self._realloc(self._memory, self._alloc_len, alloc_len)
        self._alloc_len = alloc_len
        self._data_len = data_len
      raise


  def fromfile(self, file, num):
    size = num * self.itemsize
    data = file.read(size)

    l = len(str)
    data_len = self._data_len
    self._data_len += l / self.itemsize
    self.alloc(self._data_len)
    alloc.copy_direct(self._memory + (data_len * self.itemsize), str, l)
    
    if len(data) < size:
      raise EOFError
    return


  def fromlist(self, list):
    data_len = self._data_len
    alloc_len = self._alloc_len
    self.alloc(self._data_len + len(list))

    # Try copying the data onto the array
    try:
      for i in list:
        self._setitem(self._memory, self._data_len, i)
        #self.__setitem__(self._data_len, i)
        self._data_len += 1
    except TypeError:
      if alloc_len < self._alloc_len:
        self._memory = self._realloc(self._memory, self._alloc_len, alloc_len)
        self._alloc_len = alloc_len
        self._data_len = data_len
      raise
    return


  def fromstring(self, string):
    return self.fromlist(string)
    

  def fromunicode(self, string):
    raise NotImplementedError("Unicode not supported by extarray")


  def index(self, val):
    for i in range(0, self._data_len):
      if self.__getitem__(i) == val:
        return i


  def insert(self, ind, val):
    if ind >= self._data_len:
      raise IndexError

    # Traverse backwards, appending one item forward until item i
    r = range(ind, self._data_len - 1)
    r.reverse()

    # Append the last item on again
    self.append(self.__getitem__(self._data_len - 1))
    for i in r:
      self._setitem(self._memory, i + 1, self._getitem(self._memory, i))
      #self.__setitem__(i + 1, self.__getitem__(i))

    self._setitem(self._memory, ind, val)
    #self.__setitem__(ind, val)
    return


  def memory_lock(self, val = None):
    """Enable/disable memory (re)allocation operations.
       If memory is locked (val is True), an exception will be thrown
       whenever an operation would cause (re)allocation of memory to occur.

       Calling this function with no arguments (or any value other than True or
       False) returns the current memory lock status without changing it."""
    if val == True or val == False:
      self._lock = val
    return self._lock


  def pop(self, ind = -1):
    val = None
    if ind == -1 or ind == self._data_len - 1:
      val = self.__getitem__(self._data_len - 1)
    else:
      val = self.__getitem__(ind)
      for i in range(ind, self._data_len - 1):
        self._setitem(self._memory, i, self._getitem(self._memory, i + 1))
        #self.__setitem__(i, self.__getitem__(i + 1))

    self._data_len -= 1
    return val


  def set_memory(self, val):
    """Set the address of the memory used to store the array items, and lock
       the array's memory.  Locking only prevents (re)allocation of memory,
       and can be enabled/disabled using set_memory_lock() if desired.

       If the array has already allocated some memory and the memory lock is
       False, that memory is freed.

       Useful for fitting an extarray over existing memory allocated
       elsewhere, for example a memory-mapped file."""
    if self._memory != None and self._memory != 0 and self._lock != True:
      self._free(self._memory)
    self._memory = val
    self._lock = True
    return None

  #synchronize = staticmethod(alloc.synchronize)
  def synchronize(self):
    return alloc.synchronize()


  def __setitem__(self, ind, val):
    self._setitem(self._memory, ind, val)
    return


  def __getitem__(self, ind):
    return self._getitem(self._memory, ind)


  def __str__(self):
    if self.typecode == 'c':
      s = "extarray('c', \""
      if self._data_len > 0:
        s += self._getitem(self._memory, 0)
        for i in range(1, self._data_len):
          s += self._getitem(self._memory, i)
      s += "\")"
    else:
      s = "extarray('%c', [" % (self.typecode)
      if self._data_len > 0:
        s += str(self._getitem(self._memory, 0))
        for i in range(1, self._data_len):
          s += ", %s" % str(self._getitem(self._memory, i))
      s += "])"
    return s


  # TODO - make iterator support more robust
  def __iter__(self):
    self._iter = 0
    return self


  def next(self):
    if self._iter == self._data_len:
      raise StopIteration
    i = self._iter
    self._iter += 1
    return self.__getitem__(i)


if __name__ == '__main__':
  ea1 = extarray('i', [1, 2, 3, 4, 5])
  ea2 = extarray('c', "this is a test")
  ea3 = extarray('f', [3.14, 2.8, 7.1])

  try:
    ea4 = extarray('h', 32, True)
    print "Looks like huge pages are available"
  except MemoryError, err:
    print "No huge page support: %s" % err

  ea1[3] = 10
  for i in range(0, 5):
    print ea1[i]

  ea2[4] = 'z'
  print ea2

  ea3[1] = 6.28
  for i in range(0, 3):
    print ea3[i]

  for i in ea1:
    print "iter",i
  ea1.append(6)
  print ea1
  print ea1.buffer_info()
  ea1.extend([7, 8, 9])
  print ea1
  ea1.insert(1, 10)
  print ea1
  ea1.clear()
  print ea1

  perftest_ea()
  perftest_a()

