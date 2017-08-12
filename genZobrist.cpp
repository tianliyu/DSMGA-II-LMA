
#include <cstdlib>
#include <cstdio>
#include <ctime>

#define SIZE (1000)
#define filename "zobristkey"

int main() {

    unsigned long keys[SIZE];

    printf("zorbrist keys are %lu bits.\n", sizeof(unsigned long)*8);

    srand((long)time(NULL));

    for (int i=0; i<SIZE; ++i)
        keys[i] = ((unsigned long)(rand()) << 32) | rand();


    printf("%d keys are generated.\n", SIZE);
    printf("Writing to file: %s\n", filename);

    FILE* fp = fopen(filename, "wb");
    fwrite(keys, sizeof(unsigned long), SIZE, fp);
    fclose(fp);

    printf("done!\n");

    return 0;
}
