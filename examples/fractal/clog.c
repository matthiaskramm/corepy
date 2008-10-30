/* Copyright (c) 2006-2008 The Trustees of Indiana University.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * - Neither the Indiana University nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>

#define LOG2EA 0.44269504088896340735992
#define SQRTH 0.70710678118654752440

float MINLOGF = -88.7228391116729996;
float LOGE2F = 0.693147180559945309;
float SQRTHF = 0.707106781186547524;

float 
c_log2x( float xx )
{
  /* x is mantissa, e is exponent */
  /* Do some masking to extract from xx */
  float x = xx, y, z;
  int e = (((*(unsigned int *) &x) >> 23) & 0xff) - 0x7e;

  *(unsigned int*)&x &= 0x807fffff;
  *(unsigned int*)&x |= 0x3f000000;
  
  /* normalize */
  if (x < SQRTHF) {
    e -= 1;
    x = x + x - 1.0;
  } else {
    x = x - 1.0;
  }

  /* compute polynomial */
  z = x * x;
  y = (((((((( 7.0376836292E-2 * x
	       - 1.1514610310E-1) * x
	     + 1.1676998740E-1) * x
	    - 1.2420140846E-1) * x
	   + 1.4249322787E-1) * x
	  - 1.6668057665E-1) * x
	 + 2.0000714765E-1) * x
	- 2.4999993993E-1) * x
       + 3.3333331174E-1) * x * z;
  y += -0.5 * z;

  /* convert to log base 2 */
  z = y * LOG2EA;
  z += x * LOG2EA;
  z += y;
  z += x;
  z += (float) e;

  printf("Debug: 0x%08X\n", *(unsigned int*)&z);

  return z ;
}


#ifdef MAIN
#include <stdio.h>
#include <math.h>
int main(void) 
{
  float f = 1.0;
  int i = 0;
  for(i; i < 10; i++) {
    printf("%.9f %.9f\n", log2f(f), c_log2x(f));
    f *= 10.0;
    
  }
  return 0;
}


#endif /* MAIN */
