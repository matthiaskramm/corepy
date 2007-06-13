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
    f *= 10.0;

    printf("%.9f %.9f\n", log2f(f), c_log2x(f));
  }
  return 0;
}


#endif /* MAIN */
