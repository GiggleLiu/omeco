/* Spawns 3 spin threads that burn CPU for the whole lifetime of the child
 * command, models an in-process hidden thread pool. Parent reaps the child,
 * so all CPU time rolls up through wait4 accounting. */
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>
static volatile int done = 0;
static void *spin(void *arg) {
    volatile unsigned long x = 0;
    while (!done) { for (int i = 0; i < 100000; i++) x += i; }
    return 0;
}
int main(int argc, char **argv) {
    pthread_t t[3];
    for (int i = 0; i < 3; i++) pthread_create(&t[i], 0, spin, 0);
    pid_t pid = fork();
    if (pid == 0) { execv(argv[1], &argv[1]); _exit(127); }
    int st = 0; waitpid(pid, &st, 0);
    done = 1;
    for (int i = 0; i < 3; i++) pthread_join(t[i], 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
