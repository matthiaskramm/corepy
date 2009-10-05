// cp_fb.h 
//
// Copyright (c) 2006, Mike Acton <macton@cellperformance.com>
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
// documentation files (the "Software"), to deal in the Software without restriction, including without
// limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
// EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
// AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef CP_FB_H
#define CP_FB_H

#if defined(__cplusplus)
extern "C" 
{
#endif

#include <stdint.h>
  // typedef unsigned int uint32_t;
  // typedef unsigned int* uintptr_t;

typedef struct cp_fb cp_fb;

struct cp_fb
{
    uint32_t  w;
    uint32_t  h;
    uint32_t  stride;
    uintptr_t start_addr;
    uintptr_t draw_addr[2];
    uint32_t  size;
    int       fd;
};

int  cp_fb_open( cp_fb* const fb );
void cp_fb_close( const cp_fb* const fb );
void cp_fb_wait_vsync( cp_fb* const fb );
void cp_fb_flip( cp_fb* const fb, unsigned long field_ndx );

void cp_write_pixel( cp_fb* const fb, unsigned long field_ndx, uint32_t offset, uint32_t pixel);
void cp_write_pixel_rgba( cp_fb* const fb, unsigned long field_ndx, uint32_t offset, uint32_t r, uint32_t g, uint32_t b, uint32_t a);

#if defined(__cplusplus)
}
#endif

#endif /* CP_FB_H */

