/********************************
  Author: Tian-Li Yu
  E-mail: tianliyu@nextumi.com
********************************/

extern void init_genrand (unsigned long s);
extern void init_by_array (unsigned long init_key[], int key_length);
extern unsigned long genrand_int32 (void);
extern double genrand_real1 (void); //[0,1]
extern double genrand_real2 (void); //[0,1)
extern double genrand_real3 (void); //(0,1)
extern double genrand_res53 (void); //[0,1)
