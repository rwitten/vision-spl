/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "svm_struct_latent_api_types.h"
#include "svm_struct_latent_api.h"
#include "./SFMT-src-1.3.3/SFMT.h"

#define MAX_INPUT_LINE_LENGTH 10000
#define DELTA 1
#define BASE_DIR "/afs/cs.stanford.edu/u/rwitten/scratch/temp/spm/data/"
#define CONST_FILENAME_PART "_spquantized_1000_"
#define CONST_FILENAME_SUFFIX ".mat"

int pad_cmp(const void * a, const void * b) {
  POINT_AND_DESCRIPTOR * pad_a = (POINT_AND_DESCRIPTOR *)a;
  POINT_AND_DESCRIPTOR * pad_b = (POINT_AND_DESCRIPTOR *)b;
  if (pad_a->x != pad_b->x) {
    return pad_a->x - pad_b->x;
  } else if (pad_a->y != pad_b->y) {
    return pad_a->y - pad_b->y;
  } else {
    return pad_a->descriptor - pad_b->descriptor;
  }
}

int get_sample_size(char * file) {
  int sample_size;
  FILE * fp = fopen(file, "r");
  fscanf(fp, "%d\n", &sample_size);
  fclose(fp);
  return sample_size;
}

IMAGE_KERNEL_CACHE ** init_cached_images(STRUCTMODEL * sm) {
  return (IMAGE_KERNEL_CACHE **)calloc(sm->n, sizeof(IMAGE_KERNEL_CACHE *));
}

void free_cached_images(IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm) {
  int i, k;
  for (i = 0; i < sm->n; ++i) {
    if (cached_images[i] != NULL) {
      for (k = 0; k < sm->num_kernels; ++k) {
	free(cached_images[i][k].points_and_descriptors);
      }
    }
    free(cached_images[i]);
  }
  free(cached_images);
}

SAMPLE read_struct_examples(char *file, STRUCTMODEL * sm, STRUCT_LEARN_PARM *sparm) {
  /*
    Gets and stores image file name, line number (i.e. index), label, width, and height for each example.
    Width and height should be in units such that width * height = number of options for h.
  */

  SAMPLE sample;
  int num_examples,label,height,width;
	int i;
  FILE *fp;
  char line[MAX_INPUT_LINE_LENGTH]; 
  char *pchar, *last_pchar;

  fp = fopen(file,"r");
  if (fp==NULL) {
    printf("Cannot open input file %s!\n", file);
	exit(1);
  }
  fgets(line, MAX_INPUT_LINE_LENGTH, fp);
  num_examples = atoi(line);
  sample.n = num_examples;
  sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*num_examples);
  
  for (i=0;(!feof(fp))&&(i<num_examples);i++) {
    fgets(line, MAX_INPUT_LINE_LENGTH, fp);

    //printf("%s\n", line);

    pchar = line;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    strcpy(sample.examples[i].x.image_path, line);
    pchar++;

    /* label: {0, 1} */
    last_pchar = pchar;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    label = atoi(last_pchar);
    pchar++;

    last_pchar = pchar;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    height = atoi(last_pchar);
    pchar++;

    last_pchar = pchar;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    width = atoi(last_pchar);
    pchar++;

    last_pchar = pchar;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    sample.examples[i].x.bbox_height = atoi(last_pchar);
    pchar++;

    last_pchar = pchar;
    while ((*pchar)!='\n') pchar++;
    *pchar = '\0';
    sample.examples[i].x.bbox_width = atoi(last_pchar);

    if (!label) {
      sample.examples[i].x.bbox_width = sm->bbox_width;
      sample.examples[i].x.bbox_height = sm->bbox_height;
    }
    
    if (sm->bbox_scale >= 0.0) {
      sample.examples[i].x.bbox_width = (int)(sm->bbox_scale * sample.examples[i].x.bbox_width);
      sample.examples[i].x.bbox_height = (int)(sm->bbox_scale * sample.examples[i].x.bbox_height);
    } else {
      sample.examples[i].x.bbox_width = width - 1;
      sample.examples[i].x.bbox_height = height - 1;
    }

    assert(label >= 0 && label < sparm->n_classes);
    sample.examples[i].y.label = label;
    sample.examples[i].x.width = get_num_bbox_positions(width, sample.examples[i].x.bbox_width, sm->bbox_step_x);
    sample.examples[i].x.height = get_num_bbox_positions(height, sample.examples[i].x.bbox_height, sm->bbox_step_y);
    sample.examples[i].x.example_id = i;
    sample.examples[i].x.example_cost = (label ? sparm->pos_neg_cost_ratio : 1.0);
    sample.examples[i].x.descriptor_top_left_xs = (int*)calloc(sm->num_kernels, sizeof(int));
    sample.examples[i].x.descriptor_top_left_ys = (int*)calloc(sm->num_kernels, sizeof(int));
    sample.examples[i].x.descriptor_num_acrosses = (int*)calloc(sm->num_kernels, sizeof(int));
    sample.examples[i].x.descriptor_num_downs = (int*)calloc(sm->num_kernels, sizeof(int));
  }
  assert(i==num_examples);
  fclose(fp);  
  return(sample); 
}

int get_num_bbox_positions(int image_length, int bbox_length, int bbox_step_length) {
  if (bbox_length >= image_length) return 1;
  return (int)ceil((1.0 * image_length - 1.0 * bbox_length) / (1.0 * bbox_step_length));
}

//file format is "<number of kernels>\n<kernel 0 name>\n<kernel 0 size>\n<kernel 1 name>\n...."
void read_kernel_info(char * kernel_info_file, STRUCTMODEL * sm) {
  int k;
  FILE * fp = fopen(kernel_info_file, "r");
  fscanf(fp, "%d\n", &(sm->num_kernels));
  sm->kernel_names = (char**)malloc(sm->num_kernels * sizeof(char*));
  sm->kernel_sizes = (int*)calloc(sm->num_kernels, sizeof(int));
  sm->descriptor_spacing_ys = (int*)calloc(sm->num_kernels, sizeof(int));
  sm->descriptor_spacing_xs = (int*)calloc(sm->num_kernels, sizeof(int));
  char cur_kernel_name[1024]; //if you need more than 1023 characters to name a kernel, you need help
  for (k = 0; k < sm->num_kernels; ++k) {
    assert(!feof(fp));
    fscanf(fp, "%s\n", cur_kernel_name);
    sm->kernel_names[k] = strdup(cur_kernel_name);
    fscanf(fp, "%d\n", &(sm->kernel_sizes[k]));
    fscanf(fp, "%d\n", &(sm->descriptor_spacing_ys[k]));
    fscanf(fp, "%d\n", &(sm->descriptor_spacing_xs[k]));
  }
  sm->sizePsi = 0;
  for (k = 0; k < sm->num_kernels; ++k) {
    sm->sizePsi += sm->kernel_sizes[k];
  }
}

void init_struct_model(int sample_size, char * kernel_info_file, STRUCTMODEL *sm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the dimension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

  read_kernel_info(kernel_info_file, sm);

  sm->n = sample_size;
}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
	
  int i;
  /* initialize the RNG */
	init_gen_rand(sparm->rng_seed);

	for (i=0;i<sample->n;i++) {
		sample->examples[i].h.position_x = (long) floor(genrand_res53()*(sample->examples[i].x.width-1));
		sample->examples[i].h.position_y = (long) floor(genrand_res53()*(sample->examples[i].x.height-1));
		if(sample->examples[i].h.position_x < 0 || sample->examples[i].h.position_x >= sample->examples[i].x.width-1)
			sample->examples[i].h.position_x = (long) 0;
		if(sample->examples[i].h.position_y < 0 || sample->examples[i].h.position_y >= sample->examples[i].x.height-1)
			sample->examples[i].h.position_y = (long) 0;
	}
}

/*keep this around for debugging purposes*/
//int in_bounding_box(int pixel_x, int pixel_y, LATENT_VAR h, STRUCTMODEL * sm) {
//  int bbox_start_x = h.position_x * sm->bbox_step_x;
//  int bbox_start_y = h.position_y * sm->bbox_step_y;
//  int bbox_end_x = bbox_start_x + sm->bbox_width;
//  int bbox_end_y = bbox_start_y + sm->bbox_height;
//  return (pixel_x >= bbox_start_x) && (pixel_y >= bbox_start_y) && (pixel_x < bbox_end_x) && (pixel_y < bbox_end_y);
//}

int bbox_coord_to_pixel_coord(int bbox_coord, int bbox_step) {
  return bbox_coord * bbox_step;
}

int pixel_coord_to_descriptor_coord(int pixel_coord, int descriptor_tl_offset, int descriptor_spacing) {
  double raw_descriptor_coord = ((double)pixel_coord - (double)descriptor_tl_offset) / ((double)descriptor_spacing);
  if (raw_descriptor_coord < 0.0) {
    return 0;
  } else {
    return (int)ceil(raw_descriptor_coord);
  }
}

//if the contents of files are ever cached, this would be a good place to implement that cacheing
FILE * open_kernelized_image_file(PATTERN x, int kernel_ind, STRUCTMODEL * sm) {
  char file_path[1024];
  strcpy(file_path, BASE_DIR);
  strcat(file_path, sm->kernel_names[kernel_ind]);
  strcat(file_path, "/");
  strcat(file_path, x.image_path);
  strcat(file_path, CONST_FILENAME_PART);
  strcat(file_path, sm->kernel_names[kernel_ind]);
  strcat(file_path, CONST_FILENAME_SUFFIX);
  //printf("file_path = %s\n", file_path);
  FILE * fp = fopen(file_path, "r");
  assert(fp != NULL);
  return fp;
}

//int point_cmp(const void * a, const void * b) {
//  POINT_AND_DESCRIPTOR * pad_a = (POINT_AND_DESCRIPTOR *)a;
//  POINT_AND_DESCRIPTOR * pad_b = (POINT_AND_DESCRIPTOR *)b;
//  if (pad_a->x != pad_b->x) {
//    if (pad_a->x > pad_b->x) {
//      return 1;
//    }
//    if (pad_a->x < pad_b->x) {
//      return -1;
//    }
//  } else if (pad_a->y != pad_b->y) {
//    if (pad_a->y > pad_b->y) {
//      return 1;
//    }
//    if (pad_a->y < pad_b->y) {
//      return -1;
//    }
//  } else if (pad_a->descriptor != pad_b->descriptor) { 
//I don't actually care how the descriptors are ordered, but I'm too lazy to figure out whether qsort() will think a and b are interchangeable if point_cmp returns 0 - much easier to just take the paranoid approach and only return 0 if they're actually interchangeable!
//    if (pad_a->descriptor > pad_b->descriptor) {
//      return 1;
//    }
//    if (pad_a->descriptor < pad_b->descriptor) {
//      return -1;
//    }
//  }
//  return 0;
//}

//void store_x_begins(IMAGE_KERNEL_CACHE * ikc) {
//  int p, q;
//  int * temp_index_list = (int *)malloc(ikc->num_points * sizeof(int));
//  int cur_num_x_vals = 0;
//  int cur_x_val = -1;
//  for (p = 0;  p < ikc->num_points; ++p) {
//    if (ikc->points_and_descriptors[p].x != cur_x_val) {
//      temp_index_list[cur_num_x_vals] = p;
//      cur_num_x_vals++;
//     cur_x_val = ikc->points_and_descriptors[p].x
//	}
//  }
//  ikc->num_unique_x_vals = cur_num_x_vals;
//  ikc->x_begin_indices = (int *)malloc(cur_num_x_vals * sizeof(int));
//  ikc->x_begin_pads = (POINT_AND_DESCRIPTOR *)malloc(cur_num_x_vals * sizeof(POINT_AND_DESCRIPTOR));
//  for (q = 0; q < cur_num_x_vals; ++q) {
//   ikc->x_begin_indices[q] = temp_index_list[q];
//    ikc->x_begin_pads[q] = ikc->points_and_descriptors[temp_index_list[q]];
//  }
//  free(temp_index_list);
//}

void cut_off_last_column(IMAGE_KERNEL_CACHE * ikc) {
  int p;
  int last_p = -1;
  int last_x = ikc->points_and_descriptors[ikc->num_points - 1].x;
  for (p = ikc->num_points - 1; p >= 0; --p) {
    if (ikc->points_and_descriptors[p].x != last_x) {
      last_p = p + 1;
      break;
    }
  }
  assert(last_p != -1);
  ikc->points_and_descriptors = (POINT_AND_DESCRIPTOR *)realloc(ikc->points_and_descriptors, last_p * sizeof(POINT_AND_DESCRIPTOR));
  ikc->num_points = last_p;
}

void fill_image_kernel_cache(PATTERN x, int kernel_ind, IMAGE_KERNEL_CACHE * ikc, STRUCTMODEL * sm) {
  int p;
  char throwaway_line[1024];
  FILE * fp = open_kernelized_image_file(x, kernel_ind, sm);
  fscanf(fp, "%d\n", &(ikc->num_points));
  ikc->points_and_descriptors = (POINT_AND_DESCRIPTOR *)calloc(ikc->num_points, sizeof(POINT_AND_DESCRIPTOR));
  fscanf(fp, "%s\n", throwaway_line);
  for (p = 0; p < ikc->num_points; ++p) {
    fscanf(fp, "(%d,%d):%d\n", &(ikc->points_and_descriptors[p].y), &(ikc->points_and_descriptors[p].x), &(ikc->points_and_descriptors[p].descriptor));
    assert(ikc->points_and_descriptors[p].x > 0);
    assert(ikc->points_and_descriptors[p].y > 0);
  }
  fclose(fp);
 
  /*this will sort points by x, and within that, by y*/
  qsort(ikc->points_and_descriptors, ikc->num_points, sizeof(POINT_AND_DESCRIPTOR), pad_cmp);
  
  x.descriptor_top_left_xs[kernel_ind] = ikc->points_and_descriptors[0].x;
  x.descriptor_top_left_ys[kernel_ind] = ikc->points_and_descriptors[0].y;
  
  /*Need to cut off last column because of stupid honeycomb nonsense that some idiot from the Netherlands decided to do.*/
    cut_off_last_column(ikc);

  /*and now we rely heavily on the assumption that there's grid structure in order to figure out how many descriptor points we have down and across*/
  p = 1;
  while (1) {
    int prev_y = ikc->points_and_descriptors[p - 1].y;
    int cur_y = ikc->points_and_descriptors[p].y;
    if (cur_y < prev_y) {
      break;
    }
    p++;
  }
  x.descriptor_num_downs[kernel_ind] = p;
  if ((ikc->num_points % p) != 0) {
    printf("ERROR: Something's wrong with the grid structure of the data (or Kevin's code).  p = %d, num_points = %d\n", p, ikc->num_points);
    int q;
    for (q = 0; q < ikc->num_points; ++q) {
      printf("pad.x = %d, pad.y = %d\n", ikc->points_and_descriptors[q].x,  ikc->points_and_descriptors[q].y);
    }
  }
  assert((ikc->num_points % p) == 0);
  x.descriptor_num_acrosses[kernel_ind] = ikc->num_points / p;
}

void try_cache_image(PATTERN x, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm) {
  int k;
  if (cached_images[x.example_id] == NULL) {
    printf("$"); fflush(stdout);
    cached_images[x.example_id] = (IMAGE_KERNEL_CACHE *)malloc(sm->num_kernels * sizeof(IMAGE_KERNEL_CACHE));
    IMAGE_KERNEL_CACHE * kernel_caches_for_image = cached_images[x.example_id];
    for (k = 0; k < sm->num_kernels; ++k) {
      fill_image_kernel_cache(x, k, &(kernel_caches_for_image[k]), sm);
    }
  }
}

//void descriptor_counts_to_max_pool(double * max_pool_segment, int * descriptor_counts, int kernel_size) {
//  int l;
//  double sum = 0.0;
//  for (l = 0; l < kernel_size; ++l) {
//    if (descriptor_counts[l] > 0 && max_pool_segment[l] < 1.0) {
//      max_pool_segment[l] = 1.0;
//      sum += 1.0;
//    }
//  }
//  if (sum < 1.0) return;
//  for (l = 0; l < kernel_size; ++l) {
//    max_pool_segment[l] /= sum;
//  }
//} 

int min(int a, int b) {
  if (a < b) return a;
  return b;
}

void fill_max_pool(PATTERN x, LATENT_VAR h, int kernel_ind, IMAGE_KERNEL_CACHE ** cached_images, WORD * words, int descriptor_offset, int * num_words, STRUCTMODEL * sm) {
  int cur_bbox_start_x_pixel = bbox_coord_to_pixel_coord(h.position_x, sm->bbox_step_x);
  int bbox_start_y_pixel = bbox_coord_to_pixel_coord(h.position_y, sm->bbox_step_y);
  int bbox_start_y = pixel_coord_to_descriptor_coord(bbox_start_y_pixel, x.descriptor_top_left_ys[kernel_ind], sm->descriptor_spacing_ys[kernel_ind]);
  int bbox_end_y = pixel_coord_to_descriptor_coord(bbox_start_y_pixel + x.bbox_height, x.descriptor_top_left_ys[kernel_ind], sm->descriptor_spacing_ys[kernel_ind]);
  int cur_bbox_start_x = pixel_coord_to_descriptor_coord(cur_bbox_start_x_pixel, x.descriptor_top_left_xs[kernel_ind], sm->descriptor_spacing_xs[kernel_ind]);
  int cur_bbox_end_x = pixel_coord_to_descriptor_coord(cur_bbox_start_x_pixel + x.bbox_width, x.descriptor_top_left_xs[kernel_ind], sm->descriptor_spacing_xs[kernel_ind]);
  POINT_AND_DESCRIPTOR * points_and_descriptors = cached_images[x.example_id][kernel_ind].points_and_descriptors;
  bbox_start_y = min(bbox_start_y, x.descriptor_num_downs[kernel_ind]);
  bbox_end_y = min(bbox_end_y, x.descriptor_num_downs[kernel_ind]);
  cur_bbox_start_x = min(cur_bbox_start_x, x.descriptor_num_acrosses[kernel_ind]);
  cur_bbox_end_x = min(cur_bbox_end_x, x.descriptor_num_acrosses[kernel_ind]);
  //if (use_prev_descriptor_counts) {
  //  int prev_bbox_start_x = pixel_coord_to_descriptor_coord(cur_bbox_start_x_pixel - sm->bbox_step_x, x.descriptor_top_left_xs[kernel_ind], sm->descriptor_spacing_xs[kernel_ind]);
  //  int prev_bbox_end_x = pixel_coord_to_descriptor_coord(cur_bbox_start_x_pixel + x.bbox_width - sm->bbox_step_x, x.descriptor_top_left_xs[kernel_ind], sm->descriptor_spacing_xs[kernel_ind]);
  //  prev_bbox_start_x = min(prev_bbox_start_x, x.descriptor_num_acrosses[kernel_ind]);
  //  prev_bbox_end_x = min(prev_bbox_start_x, x.descriptor_num_acrosses[kernel_ind]);
  //  struct timeval start_time;
  //  struct timeval finish_time;
  //  gettimeofday(&start_time, NULL);
  //  int l;
    //for (l = 0; l < 1000; ++l) {
  //  assert(0);
  //    get_descriptor_counts(points_and_descriptors, prev_bbox_end_x, prev_bbox_start_x, bbox_start_y, bbox_start_y, cur_bbox_end_x - prev_bbox_end_x, cur_bbox_start_x - prev_bbox_start_x, bbox_end_y - bbox_start_y, bbox_end_y - bbox_start_y, x.descriptor_num_downs[kernel_ind], descriptor_counts, kernel_ind, sm);
      //}
  //  gettimeofday(&finish_time, NULL);
  //  int million = 1000000;
  //  int microseconds = million * (int)(finish_time.tv_sec - start_time.tv_sec) + (int)(finish_time.tv_usec - start_time.tv_usec);
    //printf("get_descriptor_counts() takes %f microseconds.\n", microseconds / 1000.0);
  //} else {
    //struct timeval start_time;
    //struct timeval finish_time;
    //gettimeofday(&start_time, NULL);
    //for (l = 0; l < 1000; ++l) {
    do_max_pooling(points_and_descriptors, cur_bbox_start_x, bbox_start_y, cur_bbox_end_x - cur_bbox_start_x, bbox_end_y - bbox_start_y, x.descriptor_num_downs[kernel_ind], kernel_ind, words, descriptor_offset, num_words, sm);
    //}
    //gettimeofday(&finish_time, NULL);
    //int million = 1000000;
    //int microseconds = million * (int)(finish_time.tv_sec - start_time.tv_sec) + (int)(finish_time.tv_usec - start_time.tv_usec);
    //printf("get_descriptor_counts_entire_bbox() takes %f microseconds.\n", microseconds / 1000.0);
//  }
}

//void get_descriptor_counts(POINT_AND_DESCRIPTOR * points_and_descriptors, int add_start_x, int subtract_start_x, int add_start_y, int subtract_start_y, int add_num_across, int subtract_num_across, int add_num_down, int subtract_num_down, int total_num_down, int * descriptor_counts, int kernel_ind, STRUCTMODEL * sm) {
//  int x, y, descriptor;
  //printf("total_num_down = %d\n", total_num_down);
  //printf("add_num_across = %d\n", add_num_across);
  //printf("add_num_down = %d\n", add_num_down);
  //printf("subtract_num_across = %d\n", subtract_num_across);
  //printf("subtract_num_down = %d\n", subtract_num_down);
//  for (x = 0; x < add_num_across; ++x) {
//    for (y = 0; y < add_num_down; ++y) {
//      descriptor = points_and_descriptors[total_num_down * (x + add_start_x) + (y + add_start_y)].descriptor;
//      descriptor_counts[descriptor - 1] += 1;
//    }
//  }
//  for (x = 0; x < subtract_num_across; ++x) {
//    for (y = 0; y < subtract_num_down; ++y) {
//      descriptor = points_and_descriptors[total_num_down * (x + subtract_start_x) + (y + subtract_start_y)].descriptor;
//      descriptor_counts[descriptor - 1] -= 1;
//    }
//  }
//}

void do_max_pooling(POINT_AND_DESCRIPTOR * points_and_descriptors, int start_x, int start_y, int num_across, int num_down, int total_num_down, int kernel_ind, WORD * words, int descriptor_offset, int * num_words, STRUCTMODEL * sm) {
  int x, y, descriptor,l;
  int init_num_words = *num_words;
  double sum = 0.0;
  char * max_pool = (char*)calloc(sm->kernel_sizes[kernel_ind], sizeof(char));
  for (x = 0; x < num_across; ++x) {
    for (y = 0; y < num_down; ++y) {
      descriptor = points_and_descriptors[total_num_down * (x + start_x) + (y + start_y)].descriptor;
      if (max_pool[descriptor - 1] == '\0') { //CAUTION: Do NOT use > or < here!  char might be signed, in which case (char)0xff < '\0'!  == is fine, because even in two's complement form, (char)0xff != '\0'.
	max_pool[descriptor - 1] = (char)0xff;
	words[*num_words].wnum = descriptor - 1 + descriptor_offset;
	words[*num_words].weight = 1.0;
	sum += 1.0;
	(*num_words)++;
      }
    }
  }
  for (l = init_num_words; l < *num_words; ++l) {
    words[l].weight /= sum;
  }
  free(max_pool);
}

void zero_svector_parts(int * valid_kernels, SVECTOR * fvec, STRUCTMODEL * sm) {
  int word_ind = 0;
  int current_kernel_ind = 0;
  int kernel_start = 0;
  int kernel_cutoff = kernel_start + sm->kernel_sizes[current_kernel_ind];
  while (fvec->words[word_ind].wnum != 0) {
    assert(current_kernel_ind < sm->num_kernels);
    int index = fvec->words[word_ind].wnum - 1;
    if (index >= kernel_cutoff) {
      current_kernel_ind++;
      kernel_start = kernel_cutoff;
      kernel_cutoff = kernel_start + sm->kernel_sizes[current_kernel_ind];
    } else {
      fvec->words[word_ind].weight *= valid_kernels[current_kernel_ind];
      word_ind++;
    }
  }
}

void log_psi(PATTERN x, LABEL y, LATENT_VAR h, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, FILE * fp, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  SVECTOR * psi_vect = psi(x, y, h, cached_images, valid_kernels, sm, sparm);
  double * dense_vect = calloc(sm->sizePsi, sizeof(double));
  char img_num_str[1024];
  char * img_num_ptr = img_num_str;
  strcpy(img_num_ptr, x.image_path);
  img_num_ptr = strchr(img_num_ptr, (int)('/'));
  img_num_ptr++;
  img_num_ptr = strchr(img_num_ptr, (int)('/'));
  img_num_ptr++;
  fprintf(fp, "%s ", img_num_ptr);
  int i;
  for (i = 0; psi_vect->words[i].wnum != 0; ++i) {
    dense_vect[psi_vect->words[i].wnum - 1] = psi_vect->words[i].weight;
  }
  for (i = 0; i < sm->sizePsi; ++i) {
    fprintf(fp, "%.16g ", dense_vect[i]);
  }
  fprintf(fp, "\n");
  free_svector(psi_vect);
  free(dense_vect);
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
  assert(sparm->n_classes == 2); //if this assertion fails, you NEED to change how the previous bounding-box is used!!!
  //struct timeval start_time;
  //struct timeval finish_time;
  //gettimeofday(&start_time, NULL);

  try_cache_image(x, cached_images, sm);

  SVECTOR * fvec = NULL;
  //binary labelling for now - 1 means there's a car, 0 means there's no car
  if (y.label) {
    int num_words = 0;
    WORD * words = (WORD *)calloc(sm->sizePsi + 1, sizeof(WORD));
    double * max_pool = (double *)calloc(sm->sizePsi + 1, sizeof(double));
    int k;
    int start_ind = 1;
    for (k = 0; k < sm->num_kernels; ++k) {
      if (valid_kernels[k]) { 
        fill_max_pool(x, h, k, cached_images, words, start_ind, &num_words, sm);
      }
      start_ind += sm->kernel_sizes[k];
    }
    words[num_words].wnum = 0;
    words = (WORD *)realloc(words, (num_words + 1) * sizeof(WORD));
    fvec = create_svector_shallow(words, strdup(""), 1.0);
    free(max_pool);
    return fvec;
  } else {
    WORD * words = (WORD *)calloc(1, sizeof(WORD));
    fvec = create_svector_shallow(words, strdup(""), 1.0);
    return fvec;
  }
  
  //gettimeofday(&finish_time, NULL);

  //if (y.label) {
  //  int million = 1000000;
  //  int microseconds = million * (int)(finish_time.tv_sec - start_time.tv_sec) + (int)(finish_time.tv_usec - start_time.tv_usec);
    //    printf("psi() took %d microseconds.\n", microseconds);
  //}
  
  //  struct timeval start_time;
  //struct timeval finish_time;
  //gettimeofday(&start_time, NULL);
  //int l;
  //for (l = 0; l < 1000; ++l) {

  //}
  //gettimeofday(&finish_time, NULL);
  //int million = 1000000;
  //int microseconds = million * (int)(finish_time.tv_sec - start_time.tv_sec) + (int)(finish_time.tv_usec - start_time.tv_usec);
  //printf("create_svector_n() takes %f microseconds.\n", microseconds / 1000.0);
  

  //free(max_pool);
  //return fvec;
}

double compute_w_T_psi(PATTERN *x, int position_x, int position_y, int classi, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  double w_T_psi;
  LABEL y;
  LATENT_VAR h;
  y.label = classi;
  h.position_x = position_x;
  h.position_y = position_y;
  SVECTOR * psi_vect = psi(*x, y, h, cached_images, valid_kernels, sm, sparm);
  w_T_psi = sprod_ns(sm->w, psi_vect);
  free_svector(psi_vect);
  return w_T_psi;
}

double classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int impute, double * max_score_positive, LATENT_VAR * argmax_h_positive) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
  
//  printf("sparm->n_classes = %d, x.width = %d, x.height = %d\n", sparm->n_classes, x.width, x.height);
   int l;
	int width = x.width;
	int height = x.height;
	int cur_class, cur_position_x, cur_position_y;
	double max_score;
	double score;

        int * valid_kernels = (int*)calloc(sm->num_kernels, sizeof(int));
        for (l = 0; l < sm->num_kernels; ++l) {
          valid_kernels[l] = 1;
        }

	max_score = -DBL_MAX;
        *max_score_positive = -DBL_MAX;
	for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {
		for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
			for(cur_class = 0; cur_class < sparm->n_classes; cur_class++) {
              if(!impute) {
                  cur_position_x = h->position_x;
                  cur_position_y = h->position_y;
              }
			  score = compute_w_T_psi(&x, cur_position_x, cur_position_y, cur_class, cached_images, valid_kernels, sm, sparm);
				if(score > max_score) {
					max_score = score;
					y->label = cur_class;
					h->position_x = cur_position_x;
					h->position_y = cur_position_y;
				}
                                //printf("score = %f\n", score);
                                if (cur_class > 0 && score > *max_score_positive) {
                                  *max_score_positive = score;
                                  argmax_h_positive->position_x = cur_position_x;
                                  argmax_h_positive->position_y = cur_position_y;
                                }
			}
            if(!impute)
                break;
		}
        if(!impute)
            break;
	}

        

        free(valid_kernels);

    //printf("%d %d\n",h->position_x,h->position_y);
        
        //printf("max_score_positive = %f\n", *max_score_positive);

	return max_score;
}

void initialize_most_violated_constraint_search(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, double * max_score, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  hbar->position_x = hstar.position_x;
  hbar->position_y = hstar.position_y;
  ybar->label = y.label;
  *max_score = compute_w_T_psi(&x, hbar->position_x, hbar->position_y, ybar->label, cached_images, valid_kernels, sm, sparm);
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
//  printf("width = %d, height = %d\n", x.width, x.height);

//  time_t start_time = time(NULL);

  struct timeval start_time;
  struct timeval finish_time;
  gettimeofday(&start_time, NULL);

	int width = x.width;
	int height = x.height;
	int cur_class, cur_position_x, cur_position_y;
	double max_score,score;
	
	//make explicit the idea that (y, hstar) is what's returned if the constraint is not violated
	initialize_most_violated_constraint_search(x, hstar, y, ybar, hbar, &max_score, cached_images, valid_kernels, sm, sparm);
	
	for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {
		for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
			for(cur_class = 0; cur_class < sparm->n_classes; cur_class++) {
			  score = compute_w_T_psi(&x, cur_position_x, cur_position_y, cur_class, cached_images, valid_kernels, sm, sparm);
				if(cur_class != y.label)
					score += 1;
				if(score > max_score) {
					max_score = score;
					ybar->label = cur_class;
					hbar->position_x = cur_position_x;
					hbar->position_y = cur_position_y;
				}
			}
		}
	}

	gettimeofday(&finish_time, NULL);

	//if (y.label) {
	  //int million = 1000000;
	  //int microseconds = million * (int)(finish_time.tv_sec - start_time.tv_sec) + (int)(finish_time.tv_usec - start_time.tv_usec);
	  // printf("find_most_violated_constraint_marginrescaling() took %f milliseconds.\n", microseconds / 1000.0);
	//}

	//time_t finish_time = time(NULL);
	//printf("find_most_violated_constraint_marginrescaling took %d seconds to do %d h values.\n", (int)finish_time - (int)start_time, x.width * x.height);
	return;

}

void find_most_violated_constraint_differenty(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  int width = x.width;
  int height = x.height;
  int cur_class, cur_position_x, cur_position_y;
  double max_score,score;

  //make explicit the idea that (y, hstar) is what's returned if the constraint is not violated
  initialize_most_violated_constraint_search(x, hstar, y, ybar, hbar, &max_score, cached_images, valid_kernels, sm, sparm);

  for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {
    for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
      for(cur_class = 0; cur_class < sparm->n_classes; cur_class++) {
	if (cur_class != y.label) {
	  score = DELTA + compute_w_T_psi(&x, cur_position_x, cur_position_y, cur_class, cached_images, valid_kernels, sm, sparm);
	  if (score > max_score) {
	    max_score = score;
	    ybar->label = cur_class;
	    hbar->position_x = cur_position_x;
	    hbar->position_y = cur_position_y;
	  }
	}
      }
    }
  }

  return;
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  //printf("width = %d, height = %d\n", x.width, x.height);
  //time_t start_time = time(NULL);

  LATENT_VAR h;

h.position_x = 0;
h.position_y = 0;
if (y.label == 0) {
    return h;
}

  int l;
	int width = x.width;
	int height = x.height;
	int cur_position_x, cur_position_y;
	double max_score, score;
//	FILE	*fp;

        int * valid_kernels = (int*)calloc(sm->num_kernels, sizeof(int));
        for (l = 0; l < sm->num_kernels; ++l) {
          valid_kernels[l] = 1;
        }

	max_score = -DBL_MAX;
	for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {
	       for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
                 score = compute_w_T_psi(&x, cur_position_x, cur_position_y, y.label, cached_images, valid_kernels, sm, sparm);
			if(score > max_score) {
				max_score = score;
				h.position_x = cur_position_x;
				h.position_y = cur_position_y;
			}
		}
	}
	
        free(valid_kernels);

	//time_t finish_time = time(NULL);

	//printf("infer_latent_variables() took %d seconds to do %d h values.\n", (int)finish_time - (int)start_time, x.width * x.height);

  return(h); 
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/ 
  if (y.label==ybar.label) {
    return(0);
  } else {
    return(1);
  }
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
  Also writes bounding-box info (before sm->w)
*/
  FILE *modelfl;
  int i;
  
  modelfl = fopen(file,"w");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for output!", file);
		exit(1);
  }
  
  fprintf(modelfl, "%d\n", sm->bbox_height);
  fprintf(modelfl, "%d\n", sm->bbox_width);
  fprintf(modelfl, "%f\n", sm->bbox_scale);
  fprintf(modelfl, "%d\n", sm->bbox_step_y);
  fprintf(modelfl, "%d\n", sm->bbox_step_x);

  for (i=1;i<sm->sizePsi+1;i++) {
    fprintf(modelfl, "%d:%.16g\n", i, sm->w[i]);
  }
  fclose(modelfl);
 
}

void read_struct_model(char *model_file, STRUCTMODEL * sm) {
/*
  Reads in the learned model parameters from file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
*/

  FILE *modelfl;
  int fnum;
  double fweight;
  
  modelfl = fopen(model_file,"r");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for input!", model_file);
	exit(1);
  }
  
  sm->w = (double*)calloc(sm->sizePsi + 1, sizeof(double));
  fscanf(modelfl, "%d\n", &(sm->bbox_height));
  fscanf(modelfl, "%d\n", &(sm->bbox_width));
  fscanf(modelfl, "%lf\n", &(sm->bbox_scale));
  fscanf(modelfl, "%d\n", &(sm->bbox_step_y));
  fscanf(modelfl, "%d\n", &(sm->bbox_step_x));
  while (!feof(modelfl)) {
    fscanf(modelfl, "%d:%lf", &fnum, &fweight);
		sm->w[fnum] = fweight;
  }

  fclose(modelfl);
}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/
  int k;
  
  free(sm.w);

  for (k = 0; k < sm.num_kernels; ++k) {
    free(sm.kernel_names[k]);
  }
  free(sm.kernel_sizes);
  free(sm.kernel_names);
}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/

}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/

} 

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/

}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
  
  /* set default */
  sparm->rng_seed = 0;
  sparm->n_classes = 2;
  sparm->pos_neg_cost_ratio = 1.0;
  sparm->C = 10000;
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
    case 'j' : i++; sparm->pos_neg_cost_ratio = atof(sparm->custom_argv[i]); break;
    case 'c' : i++; sparm->C = atof(sparm->custom_argv[i]); break;
      case 's': i++; sparm->rng_seed = atoi(sparm->custom_argv[i]); break;
      case 'n': i++; sparm->n_classes = atoi(sparm->custom_argv[i]); break;
      case 't': i++; sparm->margin_type = atoi(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
}

void copy_label(LABEL l1, LABEL *l2)
{
	l2->label = l1.label;
}

void copy_latent_var(LATENT_VAR lv1, LATENT_VAR *lv2)
{
	lv2->position_x = lv1.position_x;
	lv2->position_y = lv1.position_y;
}

void print_latent_var(PATTERN x, LATENT_VAR h, FILE *flatent)
{
  char img_num_str[1024];
  char * img_num_ptr = img_num_str;
  strcpy(img_num_ptr, x.image_path);
  img_num_ptr = strchr(img_num_ptr, (int)('/'));
  img_num_ptr++;
  img_num_ptr = strchr(img_num_ptr, (int)('/'));
  img_num_ptr++;
  fprintf(flatent,"%s %d %d ", img_num_ptr, h.position_x,h.position_y);
	fflush(flatent);
}

void read_latent_var(LATENT_VAR *h, FILE *finlatent)
{
    fscanf(finlatent,"%d%d",&h->position_x,&h->position_y);
}

void print_label(LABEL l, FILE	*flabel)
{
	fprintf(flabel,"%d ",l.label);
	fflush(flabel);
}
