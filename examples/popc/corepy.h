// Support functions for C-based CorePy programs

#define SPU_SUCCESS 0x2000
#define SPU_FAILURE 0x2001
#define SPU_READY 0x000D


inline vector unsigned int get_vector_param_1() {
  register vector unsigned int v1 asm("$3"); 
  return v1;
};

inline vector unsigned int get_vector_param_2() {
  register vector unsigned int v2 asm("$4"); 
  return v2;
};

inline vector unsigned int get_vector_param_3() {
  register vector unsigned int v3 asm("$5"); 
  return v3;
};
  //register vector unsigned int v2 asm("$4"); 
  //register vector unsigned int v3 asm("$5");
  //vector unsigned int p3 = v3; // asm("$5");

inline void spu_ready() { asm("stop 0x000D"); };
