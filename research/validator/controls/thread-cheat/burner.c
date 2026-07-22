#include <pthread.h>
#include <time.h>
static void *spin(void *arg) {
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    volatile unsigned long x = 0;
    do { for (int i = 0; i < 100000; i++) x += i;
         clock_gettime(CLOCK_MONOTONIC, &now);
    } while (now.tv_sec - start.tv_sec < 1);
    return 0;
}
int main(void) {
    pthread_t t[4];
    for (int i = 0; i < 4; i++) pthread_create(&t[i], 0, spin, 0);
    for (int i = 0; i < 4; i++) pthread_join(t[i], 0);
    return 0;
}
