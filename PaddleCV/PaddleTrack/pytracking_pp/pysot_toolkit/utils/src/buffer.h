
#ifndef __STRING_BUFFER_H
#define __STRING_BUFFER_H

// Enable MinGW secure API for _snprintf_s
#define MINGW_HAS_SECURE_API 1

#ifdef _MSC_VER
#define __INLINE __inline
#else
#define __INLINE inline
#endif

#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

typedef struct string_buffer {
	char* buffer;
	int position;
	int size;
} string_buffer;

typedef struct string_list {
	char** buffer;
	int position;
	int size;
} string_list;

#define BUFFER_INCREMENT_STEP 4096

static __INLINE string_buffer* buffer_create(int L) {
	string_buffer* B = (string_buffer*) malloc(sizeof(string_buffer));
	B->size = L;
	B->buffer = (char*) malloc(sizeof(char) * B->size);
	B->position = 0;
	return B;
}

static __INLINE void buffer_reset(string_buffer* B) {
	B->position = 0;
}

static __INLINE void buffer_destroy(string_buffer** B) {
	if (!(*B)) return;
	if ((*B)->buffer) {
		free((*B)->buffer);
		(*B)->buffer = NULL;
	}
	free((*B));
	(*B) = NULL;
}

static __INLINE char* buffer_extract(const string_buffer* B) {
	char *S = (char*) malloc(sizeof(char) * (B->position + 1));
	memcpy(S, B->buffer, B->position);
	S[B->position] = '\0';
	return S;
}

static __INLINE int buffer_size(const string_buffer* B) {
	return B->position;
}

static __INLINE void buffer_push(string_buffer* B, char C) {
	int required = 1;
	if (required > B->size - B->position) {
		B->size = B->position + BUFFER_INCREMENT_STEP;
		B->buffer = (char*) realloc(B->buffer, sizeof(char) * B->size);
	}
	B->buffer[B->position] = C;
	B->position += required;
}

static __INLINE void buffer_append(string_buffer* B, const char *format, ...) {

	int required;
	va_list args;

#if defined(__OS2__) || defined(__WINDOWS__) || defined(WIN32) || defined(_MSC_VER)

	va_start(args, format);
	required = _vscprintf(format, args) + 1;
	va_end(args);
	if (required >= B->size - B->position) {
		B->size = B->position + required + 1;
		B->buffer = (char*) realloc(B->buffer, sizeof(char) * B->size);
	}
	va_start(args, format);
	required = _vsnprintf_s(&(B->buffer[B->position]), B->size - B->position, _TRUNCATE, format, args);
	va_end(args);
	B->position += required;

#else
	va_start(args, format);
	required = vsnprintf(&(B->buffer[B->position]), B->size - B->position, format, args);
	va_end(args);
	if (required >= B->size - B->position) {
		B->size = B->position + required + 1;
		B->buffer = (char*) realloc(B->buffer, sizeof(char) * B->size);
		va_start(args, format);
		required = vsnprintf(&(B->buffer[B->position]), B->size - B->position, format, args);
		va_end(args);
	}
	B->position += required;
#endif

}

static __INLINE string_list* list_create(int L) {
	string_list* B = (string_list*) malloc(sizeof(string_list));
	B->size = L;
	B->buffer = (char**) malloc(sizeof(char*) * B->size);
	memset(B->buffer, 0, sizeof(char*) * B->size);
	B->position = 0;
	return B;
}

static __INLINE void list_reset(string_list* B) {
	int i;
	for (i = 0; i < B->position; i++) {
		if (B->buffer[i]) free(B->buffer[i]);
		B->buffer[i] = NULL;
	}
	B->position = 0;
}

static __INLINE void list_destroy(string_list **B) {
	int i;

	if (!(*B)) return;

	for (i = 0; i < (*B)->position; i++) {
		if ((*B)->buffer[i]) free((*B)->buffer[i]); (*B)->buffer[i] = NULL;
	}

	if ((*B)->buffer) {
		free((*B)->buffer); (*B)->buffer = NULL;
	}

	free((*B));
	(*B) = NULL;
}

static __INLINE char* list_get(const string_list *B, int I) {
	if (I < 0 || I >= B->position) {
		return NULL;
	} else {
		if (!B->buffer[I]) {
			return NULL;
		} else {
			char *S;
			int length = strlen(B->buffer[I]);
			S = (char*) malloc(sizeof(char) * (length + 1));
			memcpy(S, B->buffer[I], length + 1);
			return S;
		}
	}
}

static __INLINE int list_size(const string_list *B) {
	return B->position;
}

static __INLINE void list_append(string_list *B, char* S) {
	int required = 1;
	int length = strlen(S);
	if (required > B->size - B->position) {
		B->size = B->position + 16;
		B->buffer = (char**) realloc(B->buffer, sizeof(char*) * B->size);
	}
	B->buffer[B->position] = (char*) malloc(sizeof(char) * (length + 1));
	memcpy(B->buffer[B->position], S, length + 1);
	B->position += required;
}

// This version of the append does not copy the string but simply takes the control of its allocation
static __INLINE void list_append_direct(string_list *B, char* S) {
	int required = 1;
	// int length = strlen(S);
	if (required > B->size - B->position) {
		B->size = B->position + 16;
		B->buffer = (char**) realloc(B->buffer, sizeof(char*) * B->size);
	}
	B->buffer[B->position] = S;
	B->position += required;
}


#endif
