#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#define min(a, b) a <= b ? a : b
#define max(a, b) a >= b ? a : b

#ifdef _OPENMP
    #include <omp.h>
#endif

#define MOD dynamic, 4

struct Image {
    unsigned char * RGB_data;
    int width;
    int height;
    bool type;
    long size;
};

void read_image(Image &, FILE *);
void processing(Image &, long);
void write_image(Image &, FILE *);

int main(int argc, char * argv[]) {
    if (argc != 5) {
        printf("Invalid arguments. Please enter:\n %s <threads_number> <input_filename> <output_filename> <coefficient>\n", argv[0]);
        exit(1);
    }

    char * endptr;
    long th_num = strtol(argv[1], &endptr, 10);
    if (th_num < 0 || *endptr != '\0') {
        printf("Invalid number of threads.\n");
        exit(1);
    }

#ifdef _OPENMP
    omp_set_num_threads(th_num);
#endif

    float coef = strtof(argv[4], &endptr);
    if (th_num < 0 || *endptr != '\0') {
        printf("Invalid coefficient. Must be float in [0, 0.5)\n");
        exit(1);
    }

    FILE * file_in = fopen(argv[2], "rb");
    if (!file_in) {
        printf("Could not open the input file.\n");
        exit(1);
    }
    Image image;
    read_image(image, file_in);
    fclose(file_in);

    long noise = (long)(image.height) * image.width * coef;
    auto start = std::chrono::steady_clock::now();
    processing(image, noise);
    auto end = std::chrono::steady_clock::now();

    int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Time (%i tread(s)): %i ms\n", (int) th_num, elapsed);

    FILE * file_out = fopen(argv[3], "wb");
    if (!file_in) {
        printf("Could not open the output file.\n");
        exit(1);
    }
    write_image(image, file_out);
    fclose(file_in);
}

void invalid_format() {
    printf("Invalid file format.\n");
    exit(1);
}

void read_image(Image & image, FILE * file_in) {
    char p, t;
    if (fscanf(file_in, "%c%c", &p, &t) != 2) {
        invalid_format();
    }
    if (p == 'P' && t == '5') {
        image.type = false;
    } else if (p == 'P' && t == '6') {
        image.type = true;
    } else {
        invalid_format();
    }
    
    int max_value;
    if (fscanf(file_in, "\n%i %i\n%i\n", &image.width, &image.height, &max_value) != 3) {
        invalid_format();
    }
    if (max_value != 255 || image.width < 1 || image.height < 1) {
        invalid_format();
    }

    image.size = image.height * image.width;
    if (image.type) {
        image.size *= 3;
    }

    image.RGB_data = (unsigned char *)(malloc(sizeof(unsigned char) * image.size));
    if (image.RGB_data == NULL) {
        printf("Could not read the input file.\n");
        exit(1);
    }
    if (fread(image.RGB_data, sizeof(unsigned char), image.size, file_in) != image.size || ferror(file_in)) {
        printf("Could not read the input file.\n");
        exit(1);
    }
}

void processing(Image & image, long noise) {
    int old_min = 0, old_max = 255;
    if (image.type) {
        long red_channel[256] = {0};
        long green_channel[256] = {0};
        long blue_channel[256] = {0};

#pragma omp parallel for reduction(+: red_channel, green_channel, blue_channel) schedule(MOD)
        for (int i = 0; i < image.size; i+= 3) {
            ++red_channel[image.RGB_data[i]];
            ++green_channel[image.RGB_data[i + 1]];
            ++blue_channel[image.RGB_data[i + 2]];
        }

        int r_min = 0, r_max = 255,
            g_min = 0, g_max = 255,
            b_min = 0, b_max = 255;

#pragma omp parallel sections
        {	
            #pragma omp section
            {
            while (red_channel[r_min] <= noise && r_min < 255)
                red_channel[r_min + 1] += red_channel[r_min++];
            }
            #pragma omp section
            {
            while (red_channel[r_max] <= noise && r_max > 0)
                red_channel[r_max - 1] += red_channel[r_max--];
            }
            #pragma omp section
            {
            while (green_channel[g_min] <= noise && g_min < 255)
                green_channel[g_min + 1] += green_channel[g_min++];
            }
            #pragma omp section
            {
            while (green_channel[g_max] <= noise && g_max > 0)
                green_channel[g_max - 1] += green_channel[g_max--];
            }
            #pragma omp section
            {
            while (blue_channel[b_min] <= noise && b_min < 255)
                blue_channel[b_min + 1] += blue_channel[b_min++];
            }
            #pragma omp section
            {
            while (blue_channel[b_max] <= noise && b_max > 0)
                blue_channel[b_max - 1] += blue_channel[b_max--];
            }
        }

        old_min = min(r_min, min(g_min, b_min));
        old_max = max(r_max, max(g_max, b_max));
    } else {
        long channel[256] = {0};
#pragma omp parallel for reduction(+: channel) schedule(MOD)
        for (int i = 0; i < image.size; i++) {
            ++channel[image.RGB_data[i]];
        }

#pragma omp parallel sections
        {	
            #pragma omp section
            {
            while (channel[old_min] <= noise && old_min < 255)
                channel[old_min + 1] += channel[old_min++];
            }
            #pragma omp section
            {
            while (channel[old_max] <= noise && old_max > 0)
                channel[old_max - 1] += channel[old_max--];
            }
        }
    }
    int coef2 = old_max - old_min;
#pragma omp parallel for schedule(MOD)
    for (int i = 0; i < image.size; i++) {
        if (image.RGB_data[i] <= old_min) {
            image.RGB_data[i] = 0;
        } else if (image.RGB_data[i] >= old_max) {
            image.RGB_data[i] = 255;
        } else {
            image.RGB_data[i] = 255 * (image.RGB_data[i] - old_min) / coef2;
        }
    }
}

void write_image(Image & image, FILE * file_out) {
    if (image.type) {
        fprintf(file_out, "P6");
    } else {
        fprintf(file_out, "P5");
    }
    fprintf(file_out, "\n%i %i\n255\n", image.width, image.height);
    fwrite(image.RGB_data, sizeof(unsigned char), image.size, file_out);
    free(image.RGB_data);
}