
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>

#include "region.h"
#include "buffer.h"

#if defined(__OS2__) || defined(__WINDOWS__) || defined(WIN32) || defined(_MSC_VER)
#ifndef isnan
#define isnan(x) _isnan(x)
#endif
#ifndef isinf
#define isinf(x) (!_finite(x))
#endif
#ifndef inline
#define inline _inline
#endif
#endif

/* Visual Studio 2013 was first to add C99 INFINITY and NAN */
#if defined (_MSC_VER) && _MSC_VER < 1800
#define INFINITY (DBL_MAX+DBL_MAX)
#define NAN (INFINITY-INFINITY)
#define round(fp) (int)((fp) >= 0 ? (fp) + 0.5 : (fp) - 0.5)
#endif

#define PRINT_BOUNDS(B) printf("[left: %.2f, top: %.2f, right: %.2f, bottom: %.2f]\n", B.left, B.top, B.right, B.bottom)

const region_bounds region_no_bounds = { -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX };

int __flags = 0;

int region_set_flags(int mask) {

	__flags |= mask;

	return __flags;

}

int region_clear_flags(int mask) {

	__flags &= ~mask;

	return __flags;

}

int __is_valid_sequence(float* sequence, int len) {
	int i;

	for (i = 0; i < len; i++) {
		if (isnan(sequence[i])) return 0;
	}

	return 1;
}


#define MAX_URI_SCHEME 16

const char* __parse_uri_prefix(const char* buffer, region_type* type) {

	int i = 0;

	*type = EMPTY;

	for (; i < MAX_URI_SCHEME; i++) {
		if ((buffer[i] >= 'a' && buffer[i] <= 'z') || buffer[i] == '+' || buffer[i] == '.' || buffer[i] == '-') continue;

		if (buffer[i] == ':') {
			if (strncmp(buffer, "rect", i - 1) == 0)
				*type = RECTANGLE;
			else if (strncmp(buffer, "poly", i - 1) == 0)
				*type = POLYGON;
			else if (strncmp(buffer, "mask", i - 1) == 0)
				*type = MASK;
			else if (strncmp(buffer, "special", i - 1) == 0)
				*type = SPECIAL;
			return &(buffer[i + 1]);
		}

		return buffer;
	}

	return buffer;

}

region_container* __create_region(region_type type) {

	region_container* reg = (region_container*) malloc(sizeof(region_container));

	reg->type = type;

	return reg;

}

static inline const char* _str_find(const char* in, const char delimiter) {

	int i = 0;
	while (in[i] && in[i] != delimiter) {
		i++;
	}

	return (in[i] == delimiter) ? &(in[i]) + 1 : NULL;

}

int _parse_sequence(const char* buffer, float** data) {

	int i;

	float* numbers = (float*) malloc(sizeof(float) * (strlen(buffer) / 2));

	const char* pch = buffer;
	for (i = 0; ; i++) {

		if (pch) {
#if defined (_MSC_VER)
			if (tolower(pch[0]) == 'n' && tolower(pch[1]) == 'a' && tolower(pch[2]) == 'n') {
				numbers[i] = NAN;
			} else {
				numbers[i] = (float) atof(pch);
			}
#else
			numbers[i] = (float) atof(pch);
#endif
		} else
			break;

		pch = _str_find(pch, ',');
	}

	if (i > 0) {
		int j;
		*data = (float*) malloc(sizeof(float) * i);
		for (j = 0; j < i; j++) { (*data)[j] = numbers[j]; }
		free(numbers);
	} else {
		*data = NULL;
		free(numbers);
	}

	return i;
}

int region_parse(const char* buffer, region_container** region) {

	float* data = NULL;
	const char* strdata = NULL;
	int num;

	region_type prefix_type;

	// const char* tmp = buffer;

	(*region) = NULL;

	if (!buffer || !buffer[0]) {
		return 1;
	}

	strdata = __parse_uri_prefix(buffer, &prefix_type);

	num = _parse_sequence(strdata, &data);

	// If at least one of the elements is NaN, then the region cannot be parsed
	// We return special region with a default code.
	if (!__is_valid_sequence(data, num) || num == 0) {
		// Preserve legacy support: if four values are given and the fourth one is a number
		// then this number is taken as a code.
		if (num == 4 && !isnan(data[3])) {
			(*region) = region_create_special(-(int) data[3]);
		} else {
			(*region) = region_create_special(TRAX_DEFAULT_CODE);
		}
		free(data);
		return 1;
	}

	if (prefix_type == EMPTY && num > 0) {
		if (num == 1)
			prefix_type = SPECIAL;
		else if (num == 4)
			prefix_type = RECTANGLE;
		else if (num >= 6 && num % 2 == 0)
			prefix_type = POLYGON;
	}

	switch (prefix_type) {
	case SPECIAL: {
		assert(num == 1);
		(*region) = (region_container*) malloc(sizeof(region_container));
		(*region)->type = SPECIAL;
		(*region)->data.special = (int) data[0];
		free(data);
		return 1;

	}
	case RECTANGLE: {
		assert(num == 4);
		(*region) = (region_container*) malloc(sizeof(region_container));
		(*region)->type = RECTANGLE;

		(*region)->data.rectangle.x = data[0];
		(*region)->data.rectangle.y = data[1];
		(*region)->data.rectangle.width = data[2];
		(*region)->data.rectangle.height = data[3];

		free(data);
		return 1;

	}
	case POLYGON: {
		int j;

		assert(num >= 6 && num % 2 == 0);

		(*region) = (region_container*) malloc(sizeof(region_container));
		(*region)->type = POLYGON;

		(*region)->data.polygon.count = num / 2;
		(*region)->data.polygon.x = (float*) malloc(sizeof(float) * (*region)->data.polygon.count);
		(*region)->data.polygon.y = (float*) malloc(sizeof(float) * (*region)->data.polygon.count);

		for (j = 0; j < (*region)->data.polygon.count; j++) {
			(*region)->data.polygon.x[j] = data[j * 2];
			(*region)->data.polygon.y[j] = data[j * 2 + 1];
		}

		free(data);
		return 1;
	case EMPTY:
		return 1;
	case MASK:
		return 1;
	}
		/*	    case MASK: {

			    	int i;
			    	int position;
			    	int value;

			    	assert(num > 4);

					(*region) = (region_container*) malloc(sizeof(region_container));
					(*region)->type = MASK;

			    	(*region)->data.mask.x = (int) data[0];
			    	(*region)->data.mask.y = (int) data[1];
			    	(*region)->data.mask.width = (int) data[2];
			    	(*region)->data.mask.height = (int) data[3];

			    	(*region)->data.mask.data = (char*) malloc(sizeof(char) * (*region)->data.mask.width * (*region)->data.mask.height);

			    	value = 0;
			    	position = 0;

			    	for (i = 4; i < num; i++) {

			    		int count =




			    	}


			    }*/
	}

	if (data) free(data);

	return 0;
}

char* region_string(region_container* region) {

	int i;
	char* result = NULL;
	string_buffer *buffer;

	if (!region) return NULL;

	buffer = buffer_create(32);

	if (region->type == SPECIAL) {

		buffer_append(buffer, "%d", region->data.special);

	} else if (region->type == RECTANGLE) {

		buffer_append(buffer, "%.4f,%.4f,%.4f,%.4f",
		              region->data.rectangle.x, region->data.rectangle.y,
		              region->data.rectangle.width, region->data.rectangle.height);

	} else if (region->type == POLYGON) {

		for (i = 0; i < region->data.polygon.count; i++) {
			buffer_append(buffer, (i == 0 ? "%.4f,%.4f" : ",%.4f,%.4f"), region->data.polygon.x[i], region->data.polygon.y[i]);
		}
	}

	if (buffer_size(buffer) > 0)
		result = buffer_extract(buffer);
	buffer_destroy(&buffer);

	return result;
}

void region_print(FILE* out, region_container* region) {

	char* buffer = region_string(region);

	if (buffer) {
		fputs(buffer, out);
		free(buffer);
	}

}

region_container* region_convert(const region_container* region, region_type type) {

	region_container* reg = NULL;
	switch (type) {

	case RECTANGLE: {

		reg = (region_container*) malloc(sizeof(region_container));
		reg->type = type;

		switch (region->type) {
		case RECTANGLE:
			reg->data.rectangle = region->data.rectangle;
			break;
		case POLYGON: {

			float top = FLT_MAX;
			float bottom = FLT_MIN;
			float left = FLT_MAX;
			float right = FLT_MIN;
			int i;

			for (i = 0; i < region->data.polygon.count; i++) {
				top = MIN(top, region->data.polygon.y[i]);
				bottom = MAX(bottom, region->data.polygon.y[i]);
				left = MIN(left, region->data.polygon.x[i]);
				right = MAX(right, region->data.polygon.x[i]);
			}

			reg->data.rectangle.x = left;
			reg->data.rectangle.y = top;
			reg->data.rectangle.width = right - left;
			reg->data.rectangle.height = bottom - top;
			break;
		}
		case SPECIAL: {
			free(reg); reg = NULL;
			break;
		}
		default: {
			free(reg); reg = NULL;
			break;
		}
		}
		break;
	}

	case POLYGON: {

		reg = (region_container*) malloc(sizeof(region_container));
		reg->type = type;

		switch (region->type) {
		case RECTANGLE: {

			reg->data.polygon.count = 4;

			reg->data.polygon.x = (float *) malloc(sizeof(float) * reg->data.polygon.count);
			reg->data.polygon.y = (float *) malloc(sizeof(float) * reg->data.polygon.count);

			if (__flags & REGION_LEGACY_RASTERIZATION) {

				reg->data.polygon.x[0] = region->data.rectangle.x;
				reg->data.polygon.x[1] = region->data.rectangle.x + region->data.rectangle.width;
				reg->data.polygon.x[2] = region->data.rectangle.x + region->data.rectangle.width;
				reg->data.polygon.x[3] = region->data.rectangle.x;

				reg->data.polygon.y[0] = region->data.rectangle.y;
				reg->data.polygon.y[1] = region->data.rectangle.y;
				reg->data.polygon.y[2] = region->data.rectangle.y + region->data.rectangle.height;
				reg->data.polygon.y[3] = region->data.rectangle.y + region->data.rectangle.height;

			} else {

				reg->data.polygon.x[0] = region->data.rectangle.x;
				reg->data.polygon.x[1] = region->data.rectangle.x + region->data.rectangle.width - 1;
				reg->data.polygon.x[2] = region->data.rectangle.x + region->data.rectangle.width - 1;
				reg->data.polygon.x[3] = region->data.rectangle.x;

				reg->data.polygon.y[0] = region->data.rectangle.y;
				reg->data.polygon.y[1] = region->data.rectangle.y;
				reg->data.polygon.y[2] = region->data.rectangle.y + region->data.rectangle.height - 1;
				reg->data.polygon.y[3] = region->data.rectangle.y + region->data.rectangle.height - 1;

			}

			break;
		}
		case POLYGON: {

			reg->data.polygon.count = region->data.polygon.count;

			reg->data.polygon.x = (float *) malloc(sizeof(float) * region->data.polygon.count);
			reg->data.polygon.y = (float *) malloc(sizeof(float) * region->data.polygon.count);

			memcpy(reg->data.polygon.x, region->data.polygon.x, sizeof(float) * region->data.polygon.count);
			memcpy(reg->data.polygon.y, region->data.polygon.y, sizeof(float) * region->data.polygon.count);

			break;
		}
		case SPECIAL: {
			free(reg); reg = NULL;
			break;
		}
		default: {
			free(reg); reg = NULL;
			break;
		}
		}
		break;

		case SPECIAL: {
			if (region->type == SPECIAL)
				// If source is also code then just copy the value
				reg = region_create_special(region->data.special);
			else
				// All types are converted to default region
				reg = region_create_special(TRAX_DEFAULT_CODE);
			break;
		}

		default:
			break;

		}

	}

	return reg;

}

void region_release(region_container** region) {

	switch ((*region)->type) {
	case RECTANGLE:
		break;
	case POLYGON:
		free((*region)->data.polygon.x);
		free((*region)->data.polygon.y);
		(*region)->data.polygon.count = 0;
		break;
	case SPECIAL: {
		break;
	}
	case MASK:
		break;
	case EMPTY:
		break;
	}

	free(*region);

	*region = NULL;

}

region_container* region_create_special(int code) {

	region_container* reg = __create_region(SPECIAL);

	reg->data.special = code;

	return reg;

}

region_container* region_create_rectangle(float x, float y, float width, float height) {

	region_container* reg = __create_region(RECTANGLE);

	reg->data.rectangle.width = width;
	reg->data.rectangle.height = height;
	reg->data.rectangle.x = x;
	reg->data.rectangle.y = y;

	return reg;

}

region_container* region_create_polygon(int count) {

	assert(count > 0);

	{

		region_container* reg = __create_region(POLYGON);

		reg->data.polygon.count = count;
		reg->data.polygon.x = (float *) malloc(sizeof(float) * count);
		reg->data.polygon.y = (float *) malloc(sizeof(float) * count);

		return reg;

	}
}

#define MAX_MASK 10000

void free_polygon(region_polygon* polygon) {

	free(polygon->x);
	free(polygon->y);

	polygon->x = NULL;
	polygon->y = NULL;

	polygon->count = 0;

}

region_polygon* allocate_polygon(int count) {

	region_polygon* polygon = (region_polygon*) malloc(sizeof(region_polygon));

	polygon->count = count;

	polygon->x = (float*) malloc(sizeof(float) * count);
	polygon->y = (float*) malloc(sizeof(float) * count);

	memset(polygon->x, 0, sizeof(float) * count);
	memset(polygon->y, 0, sizeof(float) * count);

	return polygon;
}

region_polygon* clone_polygon(const region_polygon* polygon) {

	region_polygon* clone = allocate_polygon(polygon->count);

	memcpy(clone->x, polygon->x, sizeof(float) * polygon->count);
	memcpy(clone->y, polygon->y, sizeof(float) * polygon->count);

	return clone;
}

region_polygon* offset_polygon(const region_polygon* polygon, float x, float y) {

	int i;
	region_polygon* clone = clone_polygon(polygon);

	for (i = 0; i < clone->count; i++) {
		clone->x[i] += x;
		clone->y[i] += y;
	}

	return clone;
}

region_polygon* round_polygon(const region_polygon* polygon) {

	int i;
	region_polygon* clone = clone_polygon(polygon);

	for (i = 0; i < clone->count; i++) {
		clone->x[i] = round(clone->x[i]);
		clone->y[i] = round(clone->y[i]);
	}

	return clone;
}

int point_in_polygon(const region_polygon* polygon, float x, float y) {
	int i, j, c = 0;
	for (i = 0, j = polygon->count - 1; i < polygon->count; j = i++) {
		if ( ((polygon->y[i] > y) != (polygon->y[j] > y)) &&
		        (x < (polygon->x[j] - polygon->x[i]) * (y - polygon->y[i]) / (polygon->y[j] - polygon->y[i]) + polygon->x[i]) )
			c = !c;
	}
	return c;
}

void print_polygon(const region_polygon* polygon) {

	int i;
	printf("%d:", polygon->count);

	for (i = 0; i < polygon->count; i++) {
		printf(" (%f, %f)", polygon->x[i], polygon->y[i]);
	}

	printf("\n");

}

region_bounds compute_bounds(const region_polygon* polygon) {

	int i;
	region_bounds bounds;
	bounds.top = FLT_MAX;
	bounds.bottom = -FLT_MAX;
	bounds.left = FLT_MAX;
	bounds.right = -FLT_MAX;

	for (i = 0; i < polygon->count; i++) {
		bounds.top = MIN(bounds.top, polygon->y[i]);
		bounds.bottom = MAX(bounds.bottom, polygon->y[i]);
		bounds.left = MIN(bounds.left, polygon->x[i]);
		bounds.right = MAX(bounds.right, polygon->x[i]);
	}

	return bounds;

}

region_bounds bounds_round(region_bounds bounds) {

	bounds.top = floor(bounds.top);
	bounds.bottom = ceil(bounds.bottom);
	bounds.left = floor(bounds.left);
	bounds.right = ceil(bounds.right);

	return bounds;

}

region_bounds bounds_intersection(region_bounds a, region_bounds b) {

	region_bounds result;

	result.top = MAX(a.top, b.top);
	result.bottom = MIN(a.bottom, b.bottom);
	result.left = MAX(a.left, b.left);
	result.right = MIN(a.right, b.right);

	return result;

}

region_bounds bounds_union(region_bounds a, region_bounds b) {

	region_bounds result;

	result.top = MIN(a.top, b.top);
	result.bottom = MAX(a.bottom, b.bottom);
	result.left = MIN(a.left, b.left);
	result.right = MAX(a.right, b.right);

	return result;

}

float bounds_overlap(region_bounds a, region_bounds b) {

	region_bounds rintersection = bounds_intersection(a, b);
	float intersection = (rintersection.right - rintersection.left) * (rintersection.bottom - rintersection.top);

	return MAX(0, intersection / (((a.right - a.left) * (a.bottom - a.top)) + ((b.right - b.left) * (b.bottom - b.top)) - intersection));

}

region_bounds region_create_bounds(float left, float top, float right, float bottom) {

	region_bounds result;

	result.top = top;
	result.bottom = bottom;
	result.left = left;
	result.right = right;

	return result;
}

region_bounds region_compute_bounds(const region_container* region) {

	region_bounds bounds;
	switch (region->type) {
	case RECTANGLE:
		if (__flags & REGION_LEGACY_RASTERIZATION) {
			bounds = region_create_bounds(region->data.rectangle.x,
			                              region->data.rectangle.y,
			                              region->data.rectangle.x + region->data.rectangle.width,
			                              region->data.rectangle.y + region->data.rectangle.height);
		} else {
			bounds = region_create_bounds(region->data.rectangle.x,
			                              region->data.rectangle.y,
			                              region->data.rectangle.x + region->data.rectangle.width - 1,
			                              region->data.rectangle.y + region->data.rectangle.height - 1);
		}
		break;
	case POLYGON: {
		bounds = compute_bounds(&(region->data.polygon));
		break;
	}
	default: {
		bounds = region_no_bounds;
		break;
	}
	}

	return bounds;

}

int rasterize_polygon(const region_polygon* polygon_input, char* mask, int width, int height) {

	int nodes, pixelY, i, j, swap;
	int sum = 0;
	region_polygon* polygon = (region_polygon*) polygon_input;

	int* nodeX = (int*) malloc(sizeof(int) * polygon->count);

	if (mask) memset(mask, 0, width * height * sizeof(char));

	if (__flags & REGION_LEGACY_RASTERIZATION) {

		/*  Loop through the rows of the image. */
		for (pixelY = 0; pixelY < height; pixelY++) {

			/*  Build a list of nodes. */
			nodes = 0;
			j = polygon->count - 1;

			for (i = 0; i < polygon->count; i++) {
				if (((polygon->y[i] < (double) pixelY) && (polygon->y[j] >= (double) pixelY)) ||
				        ((polygon->y[j] < (double) pixelY) && (polygon->y[i] >= (double) pixelY))) {
					nodeX[nodes++] = (int) (polygon->x[i] + (pixelY - polygon->y[i]) /
					                        (polygon->y[j] - polygon->y[i]) * (polygon->x[j] - polygon->x[i]));
				}
				j = i;
			}

			/* Sort the nodes, via a simple “Bubble” sort. */
			i = 0;
			while (i < nodes - 1) {
				if (nodeX[i] > nodeX[i + 1]) {
					swap = nodeX[i];
					nodeX[i] = nodeX[i + 1];
					nodeX[i + 1] = swap;
					if (i) i--;
				} else {
					i++;
				}
			}

			/*  Fill the pixels between node pairs. */
			for (i = 0; i < nodes; i += 2) {
				if (nodeX[i] >= width) break;
				if (nodeX[i + 1] > 0 ) {
					if (nodeX[i] < 0 ) nodeX[i] = 0;
					if (nodeX[i + 1] > width) nodeX[i + 1] = width - 1;
					for (j = nodeX[i]; j < nodeX[i + 1]; j++) {
						if (mask) mask[pixelY * width + j] = 1;
						sum++;
					}
				}
			}
		}

	} else {

		polygon = round_polygon(polygon_input);

		/*  Loop through the rows of the image. */
		for (pixelY = 0; pixelY < height; pixelY++) {

			/*  Build a list of nodes. */
			nodes = 0;
			j = polygon->count - 1;

			for (i = 0; i < polygon->count; i++) {
				if ((((int)polygon->y[i] <= pixelY) && ((int)polygon->y[j] > pixelY)) ||
				        (((int)polygon->y[j] <= pixelY) && ((int)polygon->y[i] > pixelY)) ||
				        (((int)polygon->y[i] < pixelY) && ((int)polygon->y[j] >= pixelY)) ||
				        (((int)polygon->y[j] < pixelY) && ((int)polygon->y[i] >= pixelY)) ||
				        (((int)polygon->y[i] == (int)polygon->y[j]) && ((int)polygon->y[i] == pixelY))) {
					double r = (polygon->y[j] - polygon->y[i]);
					double k = (polygon->x[j] - polygon->x[i]);
					if (r != 0)
						nodeX[nodes++] = (int) ((double) polygon->x[i] + (double) (pixelY - polygon->y[i]) / r * k);
				}
				j = i;
			}
			/* Sort the nodes, via a simple “Bubble” sort. */
			i = 0;
			while (i < nodes - 1) {
				if (nodeX[i] > nodeX[i + 1]) {
					swap = nodeX[i];
					nodeX[i] = nodeX[i + 1];
					nodeX[i + 1] = swap;
					if (i) i--;
				} else {
					i++;
				}
			}

			/*  Fill the pixels between node pairs. */
			i = 0;
			while (i < nodes - 1) {
				// If a point is in the line then we get two identical values
				// Ignore the first
				if (nodeX[i] == nodeX[i + 1]) {
					i++;
					continue;
				}

				if (nodeX[i] >= width) break;
				if (nodeX[i + 1] >= 0) {
					if (nodeX[i] < 0) nodeX[i] = 0;
					if (nodeX[i + 1] >= width) nodeX[i + 1] = width - 1;
					for (j = nodeX[i]; j <= nodeX[i + 1]; j++) {
						if (mask) mask[pixelY * width + j] = 1;
						sum++;
					}
				}
				i += 2;

			}
		}

		free_polygon(polygon);

	}

	free(nodeX);

	return sum;
}

float compute_polygon_overlap(const region_polygon* p1, const region_polygon* p2, float *only1, float *only2, region_bounds bounds) {

	int i;
	int vol_1 = 0;
	int vol_2 = 0;
	int mask_1 = 0;
	int mask_2 = 0;
	int mask_intersect = 0;
	char* mask1 = NULL;
	char* mask2 = NULL;
	double a1, a2;
	float x, y;
	int width, height;
	region_polygon *op1, *op2;
	region_bounds b1, b2;

	if (__flags & REGION_LEGACY_RASTERIZATION) {
		b1 = bounds_intersection(compute_bounds(p1), bounds);
		b2 = bounds_intersection(compute_bounds(p2), bounds);
	} else {
		b1 = bounds_intersection(bounds_round(compute_bounds(p1)), bounds);
		b2 = bounds_intersection(bounds_round(compute_bounds(p2)), bounds);
	}

	x = MIN(b1.left, b2.left);
	y = MIN(b1.top, b2.top);

	width = (int) (MAX(b1.right, b2.right) - x) + 1;
	height = (int) (MAX(b1.bottom, b2.bottom) - y) + 1;

	// Fixing crashes due to overflowed regions, a simple check if the ratio
	// between the two bounding boxes is simply too big and the overlap would
	// be 0 anyway.

	a1 = (b1.right - b1.left) * (b1.bottom - b1.top);
	a2 = (b2.right - b2.left) * (b2.bottom - b2.top);

	if (a1 / a2 < 1e-10 || a2 / a1 < 1e-10 || width < 1 || height < 1) {

		if (only1)
			(*only1) = 0;

		if (only2)
			(*only2) = 0;

		return 0;

	}

	if (bounds_overlap(b1, b2) == 0) {

		if (only1 || only2) {
			vol_1 = rasterize_polygon(p1, NULL, b1.right - b1.left + 1, b1.bottom - b1.top + 1);
			vol_2 = rasterize_polygon(p2, NULL, b2.right - b2.left + 1, b2.bottom - b2.top + 1);

			if (only1)
				(*only1) = (float) vol_1 / (float) (vol_1 + vol_2);

			if (only2)
				(*only2) = (float) vol_2 / (float) (vol_1 + vol_2);
		}

		return 0;

	}

	mask1 = (char*) malloc(sizeof(char) * width * height);
	mask2 = (char*) malloc(sizeof(char) * width * height);

	op1 = offset_polygon(p1, -x, -y);
	op2 = offset_polygon(p2, -x, -y);

	rasterize_polygon(op1, mask1, width, height);
	rasterize_polygon(op2, mask2, width, height);

	for (i = 0; i < width * height; i++) {
		if (mask1[i]) vol_1++;
		if (mask2[i]) vol_2++;
		if (mask1[i] && mask2[i]) mask_intersect++;
		else if (mask1[i]) mask_1++;
		else if (mask2[i]) mask_2++;
	}

	free_polygon(op1);
	free_polygon(op2);

	free(mask1);
	free(mask2);

	if (only1)
		(*only1) = (float) mask_1 / (float) (mask_1 + mask_2 + mask_intersect);

	if (only2)
		(*only2) = (float) mask_2 / (float) (mask_1 + mask_2 + mask_intersect);

	return (float) mask_intersect / (float) (mask_1 + mask_2 + mask_intersect);

}

#define COPY_POLYGON(TP, P) { P.count = TP->data.polygon.count; P.x = TP->data.polygon.x; P.y = TP->data.polygon.y; }

region_overlap region_compute_overlap(const region_container* ra, const region_container* rb, region_bounds bounds) {

	region_container* ta = (region_container *) ra;
	region_container* tb = (region_container *) rb;
	region_overlap overlap;
	overlap.overlap = 0;
	overlap.only1 = 0;
	overlap.only2 = 0;

	if (ra->type == RECTANGLE)
		ta = region_convert(ra, POLYGON);

	if (rb->type == RECTANGLE)
		tb = region_convert(rb, POLYGON);

	if (ta->type == POLYGON && tb->type == POLYGON) {

		region_polygon p1, p2;

		COPY_POLYGON(ta, p1);
		COPY_POLYGON(tb, p2);

		overlap.overlap = compute_polygon_overlap(&p1, &p2, &(overlap.only1), &(overlap.only2), bounds);

	}

	if (ta != ra)
		region_release(&ta);

	if (tb != rb)
		region_release(&tb);

	return overlap;

}

int region_contains_point(region_container* r, float x, float y) {
	
	if (r->type == RECTANGLE) {
		if (x >= (r->data.rectangle).x && x <= ((r->data.rectangle).width + (r->data.rectangle).x) &&
			y >= (r->data.rectangle).y && y <= ((r->data.rectangle).height + (r->data.rectangle).y))
            return 1;
        return 0;
	}

	if (r->type == POLYGON)
		return point_in_polygon(&(r->data.polygon), x, y);

	return 0;

}

void region_get_mask(region_container* r, char* mask, int width, int height) {

	region_container* t = r;

	if (r->type == RECTANGLE)
		t = region_convert(r, POLYGON);

	rasterize_polygon(&(t->data.polygon), mask, width, height);

	if (t != r)
		region_release(&t);

}

void region_get_mask_offset(region_container* r, char* mask, int x, int y, int width, int height) {

	region_container* t = r;
	region_polygon *p;

	if (r->type == RECTANGLE)
		t = region_convert(r, POLYGON);

	p = offset_polygon(&(t->data.polygon), -x, -y);

	rasterize_polygon(p, mask, width, height);

	free_polygon(p);

	if (t != r)
		region_release(&t);

}

