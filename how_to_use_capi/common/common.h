#ifndef __CAPI_EXAMPLE_COMMON_H__
#define __CAPI_EXAMPLE_COMMON_H__
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define CHECK(stmt)                                                      \
  do {                                                                   \
    paddle_error __err__ = stmt;                                         \
    if (__err__ != kPD_NO_ERROR) {                                       \
      fprintf(stderr, "Invoke paddle error %d in " #stmt "\n", __err__); \
      exit(__err__);                                                     \
    }                                                                    \
  } while (0)

void* read_config(const char* filename, long* size) {
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Open %s error\n", filename);
    return NULL;
  }

  struct stat filestat;
  stat(filename, &filestat);
  *size = filestat.st_size;

  void* buf = malloc(*size);
  if (buf == NULL) {
    fprintf(stderr, "Memory error\n");
    return NULL;
  }

  size_t result = fread(buf, 1, *size, file);
  if (result != *size) {
    fprintf(stderr, "Read %s error\n", filename);
    free(buf);
    return NULL;
  }

  fclose(file);
  return buf;
}
#endif