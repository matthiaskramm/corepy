/* --------------------------------------------------------------  */
/* (C)Copyright 2001,2006,                                         */
/* International Business Machines Corporation,                    */
/* Sony Computer Entertainment, Incorporated,                      */
/* Toshiba Corporation,                                            */
/*                                                                 */
/* All Rights Reserved.                                            */
/* --------------------------------------------------------------  */
/* PROLOG END TAG zYx                                              */
#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include "jaccard.h"
volatile parm_context ctx;
volatile unsigned int jaccard_array[1024];


vector unsigned int const1 = {0xFF, 0xFF, 0xFF, 0xFF};


vector unsigned char zero = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
vector unsigned char pattern1 = {16,1,16,3,16,5,16,7,16,9,16,11,16,13,16,15};
vector unsigned char pattern2 = {16,0,16,2,16,4,16,6,16,8,16,10,16,12,16,14};
vector unsigned char pattern3 = {16,16,16,16,16,16,16,16,16,16,16,16,16,16,14,15};
vector unsigned char pattern4 = {1,5,9,13,16,16,16,16,16,16,16,16,16,16,16,16};
vector unsigned char pattern5 = {3,7,11,15,16,16,16,16,16,16,16,16,16,16,16};
vector unsigned char pattern6 = {16,16,16,16,16,16,16,16,16,16,16,16,16,16,0,1};
vector unsigned char pattern7 = {16,16,16,16,16,16,16,16,16,16,16,16,16,16,2,3};
vector unsigned short short_zero = {0,0,0,0,0,0,0,0};

volatile parm_context ctx;


static inline double vec_sumb_jaccard_char( vector unsigned char vec1, vector unsigned char vec2)
{
    register vector unsigned short vec5, vec6, vec7;

    vector unsigned char temp1, temp2;
    vector unsigned char char_vec1, char_vec2, char_vec3;
    vector double a,b;

    temp1 = spu_cntb(spu_and(vec1, vec2));
    temp2 = spu_cntb(spu_xor(vec1, vec2));
    
    char_vec1 = (vector unsigned char)(spu_sumb(temp1, temp2));
    char_vec2 = (vector unsigned char)(spu_shuffle(char_vec1, zero, pattern4));
    char_vec3 = (vector unsigned char)(spu_shuffle(char_vec1,zero,pattern5));

    vec5 = spu_sumb(char_vec2, char_vec3);
    
    vec6 = spu_shuffle(vec5, short_zero,  pattern6);
    vec7 = spu_shuffle(vec5, short_zero,  pattern7);
    vec7 = spu_add(vec6, vec7);

    a = (vector double)vec6;
    b = (vector double)vec7;

    return spu_extract(a,1)/spu_extract(b,1);
}


int main(unsigned long long spu_id __attribute__ ((unused)), unsigned long long parm)
{
  int i, j;
  int left, cnt;

  unsigned int tag_id;
  double coeff;
  

  // Reserve a tag ID
  vector unsigned char vec1 = {0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55};
  vector unsigned char vec2 = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
  
  coeff = vec_sumb_jaccard_char(vec1, vec2);
  printf("Value %f\n",coeff);

  return (0);
}


