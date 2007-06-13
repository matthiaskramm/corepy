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
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "c9",
    "c10"
  };
  
  
  int i = 0;
  unsigned int *hex = (void*)f;

  for(i; i < 15; i++) {
    printf("%s: 0x%08X\n", s[i], *hex);
    hex++;
  }
  return 0;
}
