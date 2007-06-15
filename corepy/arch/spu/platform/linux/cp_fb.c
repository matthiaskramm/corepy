// cp_fb.c 
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

// NOTES:
// From Geert Uytterhoeven 2007-01-26 04:50:44, 
//      http://patchwork.ozlabs.org/linuxppc/patch?id=9143
//
//     "As the actual graphics hardware cannot be accessed directly by Linux,
//     ps3fb uses a virtual frame buffer in main memory. The actual screen image is
//     copied to graphics memory by the GPU on every vertical blank, by making a
//     hypervisor call."
//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <sys/time.h>
// #include <asm/ps3fb.h>
#include "ps3fb.h"
#include <linux/vt.h>
#include <linux/kd.h>
#include "cp_fb.h"

static inline const char*
select_error_str( int existing_error, const char* const existing_error_str, int new_error, const char* const new_error_str )
{
  // Only report the first error found - any error that follows is probably just a cascading effect.
  const char* error_str = (char*)( (~(intptr_t)existing_error & (intptr_t)new_error & (intptr_t)new_error_str)
                                 |  ((intptr_t)existing_error & (intptr_t)existing_error_str) );

  return (error_str);
}

int
cp_fb_open( cp_fb* const fb )
{
    const char*    error_str      = NULL;
    int            error          = 0;

    // Open framebuffer device

    const int   fb_fd             = open( "/dev/fb0", O_RDWR );
    const int   open_fb_error     = (fb_fd >> ((sizeof(int)<<3)-1));
    const char* open_fb_error_str = "Could not open /dev/fb0. Check permissions.";
  
    error_str = select_error_str( error, error_str, open_fb_error, open_fb_error_str );
    error     = error | open_fb_error;

    // Check for vsync

    struct fb_vblank vblank;

    const int   get_vblank_error     = ioctl(fb_fd, FBIOGET_VBLANK, (unsigned long)&vblank);
    const char* get_vblank_error_str = "Could not get vblank info (FBIOGET_VBLANK)";

    error_str = select_error_str( error, error_str, get_vblank_error, get_vblank_error_str );
    error     = error | get_vblank_error;

    const int   has_vsync            = vblank.flags & FB_VBLANK_HAVE_VSYNC;
    const int   has_vsync_error      = (~(-has_vsync|has_vsync))>>((sizeof(int)<<3)-1);
    const char* has_vsync_error_str  = "No vsync available (FB_VBLANK_HAVE_VSYNC)";

    error_str = select_error_str( error, error_str, has_vsync_error, has_vsync_error_str );
    error     = error | has_vsync_error;

    // Get screen resolution and frame count

    struct ps3fb_ioctl_res res;

    const int   screeninfo_error     = ioctl(fb_fd, PS3FB_IOCTL_SCREENINFO, (unsigned long)&res);
    const char* screeninfo_error_str = "Could not get screen info (PS3_IOCTL_SCREENINFO)";

    error_str = select_error_str( error, error_str, screeninfo_error, screeninfo_error_str );
    error     = error | screeninfo_error;

    const int   has_at_least_double_buffer           = (res.num_frames - 2) >> ((sizeof(res.num_frames)<<3)-1);
    const int   has_at_least_double_buffer_error     = ~has_at_least_double_buffer;
    const char* has_at_least_double_buffer_error_str = "Could not get screen info (PS3_IOCTL_SCREENINFO)";

    error_str = select_error_str( error, error_str, has_at_least_double_buffer_error, has_at_least_double_buffer_error_str );
    error     = error | has_at_least_double_buffer_error;

    const uint32_t bpp                      = 4; // This is fixed for PS3 fb, and there's not a test for it.
    const uint32_t frame_size               = res.xres * res.yres * bpp;
    const uint32_t double_buffer_frame_size = frame_size * 2;

    // const uint32_t frame_top_margin_size    = res.xres * res.yoff * bpp;
    // const uint32_t frame_bottom_margin_size = frame_top_margin_size;
    // const uint32_t frame_size               = frame_full_size; /* - ( frame_top_margin_size + frame_bottom_margin_size ); */
    // const uint32_t double_buffer_frame_size = frame_size * 2;

    const uintptr_t fb_addr           = (uintptr_t)mmap(NULL, double_buffer_frame_size, PROT_READ|PROT_WRITE, MAP_SHARED, fb_fd, 0);
    const int       fb_mmap_error     = fb_addr >> ((sizeof(uintptr_t)<<3)-1);
    const char*     fb_mmap_error_str = "Could not get mmap frame buffer";

    error_str = select_error_str( error, error_str, fb_mmap_error, fb_mmap_error_str );
    error     = error | fb_mmap_error;

    // Take control of frame buffer from kernel
    ioctl(fb_fd, PS3FB_IOCTL_ON, 0);

    // yoff is the number of lines that cannot be copied to the CRT before the vblank. For the most part this represents
    // unusable frame buffer space. While it is possible to draw to the area if you draw in the opposite frame buffer's
    // offset space, which will (due to poor draw timing by ps3fb) be the thing that is actually drawn, it's very 
    // difficult to work with in practice. So:
    //
    //     (1)  The y offset area will be treated as "off limits".
    //     (2)  An equivalent border will be created at the bottom, so the frame looks balanced even though it is
    //          not entirely full screen. 

    // xoff is the number of lines that cannot be copied to the CRT before the hblank.
    // Similar to the y offset space, the x offset space is displayed on the wrong (previous) line. So:
    //
    //     (1)  The x offset area will be treated as "off limits".
    //     (2)  An equivalent border will be created at the right, so the frame looks balanced even though it is
    //          not entirely full screen. 

    uintptr_t draw_start_addr = fb_addr;
    uintptr_t draw_next_addr  = draw_start_addr + ( res.yres * res.xres * bpp );
    uintptr_t drawable_h      = res.yres - ( 2 * res.yoff );
    uintptr_t drawable_w      = res.xres - ( 2 * res.xoff );

    // xoff is the number of lines that cannot be copied to the CRT before the hblank. This area is much easier to use. 
    // Similar to the y offset space, the x offset space is displayed on the wrong (previous) line. So:
    // In principle, it is possible to steal back the x offset space by shifting back the line address to the 
    // start of the border of the previous line. Like so:
    //
    //     (1)  One additional line will be taken from the height so the a complete horizontal line can be started
    //          early.
    //     (2)  The frame buffer address returned in cp_fb will be offset by (xres-xoff) in order for the remaining
    //          space to represent a rectangular area of drawable memory.
    //
    //     i.e. 
    //     uintptr_t draw_start_addr = fb_addr + ( ( res.xres - res.xoff ) * bpp );
    //     uintptr_t draw_next_addr  = draw_start_addr + ( res.yres * res.xres * bpp );
    //     uintptr_t drawable_h      = res.yres - 1 - ( 2 * res.yoff );
    //     uintptr_t drawable_w      = res.xres;
    //
    //     But I wouldn't recommend it, since on some CRTs the effect of this would be that the frame does not appear
    //     square.

    fb->stride        = res.xres;
    fb->w             = drawable_w;
    fb->h             = drawable_h;
    fb->fd            = fb_fd;
    fb->start_addr    = fb_addr;
    fb->size          = double_buffer_frame_size;
    fb->draw_addr[0]  = draw_start_addr;
    fb->draw_addr[1]  = draw_next_addr;

    // Clear out the whole buffer. Any unused space is black. It's also convinient to start with a cleared frame
    // buffer for the user.

    memset((void*)fb_addr, 0x00, double_buffer_frame_size );

    return (error);
}

void
cp_fb_close( const cp_fb* const  fb )
{
    // Give frame buffer control back to the kernel
    ioctl(fb->fd, PS3FB_IOCTL_OFF, 0);

    munmap( (void*)fb->start_addr, fb->size );

    close(fb->fd);
}

void
cp_fb_wait_vsync( cp_fb* const  fb )
{
    unsigned long crt = 0;

    ioctl(fb->fd, FBIO_WAITFORVSYNC, &crt );
}

void
cp_fb_flip( cp_fb* const  fb, unsigned long field_ndx )
{
    ioctl(fb->fd, PS3FB_IOCTL_FSEL,  &field_ndx );
}

void 
cp_write_pixel( cp_fb* const fb, unsigned long field_ndx, uint32_t offset, uint32_t pixel)
{
  *(((uint32_t*)fb->draw_addr[field_ndx]) + offset) = pixel;
}


void 
cp_write_pixel_rgba( cp_fb* const fb, unsigned long field_ndx, uint32_t offset, 
                     uint32_t r, uint32_t g, uint32_t b, uint32_t a)
{
  uint32_t rgba;

  // bgra
  rgba  = ( b & 0x000000ff );
  rgba |= ( g & 0x000000ff ) << 8;
  rgba |= ( r & 0x000000ff ) << 16;
  rgba |= ( a & 0x000000ff ) << 24;

  *(((uint32_t*)fb->draw_addr[field_ndx]) + offset) = rgba;
}
