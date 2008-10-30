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

#define LOG2EA 0.44269504088896340735992
#define SQRTH 0.70710678118654752440

float MINLOGF = -88.7228391116729996;
float LOGE2F = 0.693147180559945309;
float SQRTHF = 0.707106781186547524;

float c1 = 7.0376836292E-2;
float c2 = 1.1514610310E-1;
float c3 = 1.1676998740E-1;
float c4 = 1.2420140846E-1;
float c5 = 1.4249322787E-1;
float c6 = 1.6668057665E-1;
float c7 = 2.0000714765E-1;
float c8 = 2.4999993993E-1;
float c9 = 3.3333331174E-1;
float c10 = -0.5;          


#include <stdio.h>

int main(void) 
{
  float f[] = {
    LOG2EA, 
    SQRTH,
    MINLOGF,
    LOGE2F,
    SQRTHF,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10
  };

  char *s[] = {
    "LOG2EA", 
    "SQRTH",
    "MINLOGF",
    "LOGE2F",
    "SQRTHF",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10"
  };
  
  
  int i = 0;
  unsigned int *hex = (void*)f;
  
  int a = -126;
  printf("a: %08X\n", a);

  for(i; i < 15; i++) {
    printf("'%s': 0x%08X,  # %.10E\n", s[i], *hex, f[i]);
    hex++;
  }
  return 0;
}
