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
#include "ess.hh"

#define MAX_INPUT_LINE_LENGTH 10000
#define DELTA 1
#define BASE_DIR "/afs/cs.stanford.edu/u/rwitten/scratch/mkl_features/"
//#define BASE_DIR "/Users/rafiwitten/scratch/mkl_features/"
#define CONST_FILENAME_PART "_spquantized_1000_"
#define CONST_FILENAME_SUFFIX ".mat"
#define NUM_BBOXES_PER_IMAGE 800
#define W_SCALE ((double)1e4)

#define BASE_HEIGHT 75
#define BASE_WIDTH 125
#define X_STEP 50
#define Y_STEP 50
#define SCALE_FACTOR 1.5

int pad_cmp(const void * a, const void * b) {
  POINT_AND_DESCRIPTOR * pad_a = (POINT_AND_DESCRIPTOR *)a;
  POINT_AND_DESCRIPTOR * pad_b = (POINT_AND_DESCRIPTOR *)b;
  if (pad_a->y != pad_b->y) {
    return pad_a->y - pad_b->y;
  } else if (pad_a->x != pad_b->x) {
    return pad_a->x - pad_b->x;
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

IMAGE_KERNEL_CACHE ** init_cached_images(EXAMPLE* ex,STRUCTMODEL * sm) {
  IMAGE_KERNEL_CACHE** cached_images = (IMAGE_KERNEL_CACHE **)calloc(sm->num_distinct_images, sizeof(IMAGE_KERNEL_CACHE *));
	for(int i = 0; i < sm->n ; i++)
		try_cache_image(ex[i].x, cached_images, sm);
	return cached_images;
}

void free_cached_images(IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm) {
  int i, k;
  for (i = 0; i < sm->num_distinct_images; ++i) {
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
  int num_examples,height,width;
	int i;
  FILE *fp;
  char line[MAX_INPUT_LINE_LENGTH]; 
  char image_path[1024];
  char *pchar, *last_pchar;

  fp = fopen(file,"r");
  if (fp==NULL) {
    printf("Cannot open input file %s!\n", file);
	exit(1);
  }
  fgets(line, MAX_INPUT_LINE_LENGTH, fp);
  num_examples = atoi(line);
  sm->num_distinct_images = num_examples;
  sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*num_examples*sparm->n_classes);
  int example_ind = 0;
  for (i=0;(!feof(fp))&&(i<num_examples);i++) {
    fgets(line, MAX_INPUT_LINE_LENGTH, fp);

    //printf("%s\n", line);

    pchar = line;
    while ((*pchar) != ' ') pchar++;
    *pchar = '\0';
    strcpy(image_path, line);
    pchar++;

    last_pchar = pchar;
	while ((*pchar) != ' ') pchar++;
	*pchar = '\0';
	height = atoi(last_pchar);
	pchar++;
	
	last_pchar = pchar;
	while ((*pchar) != ' ') pchar++;
	*pchar = '\0';
	width = atoi(last_pchar);
	pchar++;

	int * correct_class_list = (int*)calloc(sparm->n_classes, sizeof(int));
	int num_correct_classes = 0;
	while (1) {
		last_pchar = pchar;
		while ((*pchar) != ' ' && (*pchar) != '\n') pchar++;
		int end_flag = (*pchar == '\n');
		*pchar = '\0';
		correct_class_list[num_correct_classes] = atoi(last_pchar);
		num_correct_classes++;
		pchar++;
		if (end_flag) break;
	}
	int * correct_class_set = (int*)calloc(sparm->n_classes, sizeof(int));
	int j;
	for (j = 0; j < num_correct_classes; ++j) {
		correct_class_set[correct_class_list[j]] = 1;
	}
	for (j = 0; j < num_correct_classes; ++j) {
		sample.examples[example_ind].y.label = correct_class_list[j];
		sample.examples[example_ind].x.also_correct = (int*)calloc(sparm->n_classes, sizeof(int));
		memcpy(sample.examples[example_ind].x.also_correct, correct_class_set, sparm->n_classes * sizeof(int));
		sample.examples[example_ind].x.also_correct[correct_class_list[j]] = 0;
		sample.examples[example_ind].x.width_pixel = width;
		sample.examples[example_ind].x.height_pixel = height;
		sample.examples[example_ind].x.example_id = example_ind;
		sample.examples[example_ind].x.image_id = i;
		strcpy(sample.examples[example_ind].x.image_path, image_path);
    		sample.examples[example_ind].x.example_cost = 1.0; //for now
  		sample.examples[example_ind].x.descriptor_top_left_xs = (int*)calloc(sm->num_kernels, sizeof(int));
    		sample.examples[example_ind].x.descriptor_top_left_ys = (int*)calloc(sm->num_kernels, sizeof(int));
    		sample.examples[example_ind].x.descriptor_num_acrosses = (int*)calloc(sm->num_kernels, sizeof(int));
    		sample.examples[example_ind].x.descriptor_num_downs = (int*)calloc(sm->num_kernels, sizeof(int));
		example_ind++;
	}
	free(correct_class_list);
	free(correct_class_set);
  }
  sample.n = example_ind;
	sample.examples = (EXAMPLE *)realloc(sample.examples, sample.n * sizeof(EXAMPLE));
	sm->n = sample.n;
  fclose(fp);  
  return(sample); 
}

int get_num_bbox_positions(int image_length, int bbox_length, int bbox_step_length) {
  if (bbox_length >= image_length) return 1;
  return (int)ceil((1.0 * image_length - 1.0 * bbox_length) / (1.0 * bbox_step_length));
}

void load_meta_kernel(STRUCTMODEL* sm, int kernel_num)
{
    sm->meta_w[kernel_num] = (double*) malloc( sizeof(double) * (sm->meta_kernel_sizes[kernel_num]));
    char filename[1024];
    printf("by the time we get here filestub is %s\n", sm->filestub);
    sprintf(filename, "%s%s.model", sm->filestub, sm->kernel_names[kernel_num]);
    int fnum;
    double fweight;
    FILE* modelfl = fopen(filename, "r");
    printf("opening filename %s\n", filename);
    assert(modelfl);
    int whatshouldbefnum=1;
    while (!feof(modelfl)) {
        fscanf(modelfl, "%d:%lf\n", &fnum, &fweight);
        assert(whatshouldbefnum==fnum);
        sm->meta_w[kernel_num][fnum-1] = fweight;
        whatshouldbefnum++;
    }
    assert(whatshouldbefnum==sm->meta_kernel_sizes[kernel_num]+1);
    fclose(modelfl);
}

//file format is "<number of kernels>\n<kernel 0 name>\n<kernel 0 size>\n<kernel 1 name>\n...."
void read_kernel_info(char * kernel_info_file, STRUCTMODEL * sm, STRUCT_LEARN_PARM* sparm) {
  int k;
  FILE * fp = fopen(kernel_info_file, "r");
  int firstline;
  assert(fp);
  fscanf(fp, "%d\n", &firstline);
    if(firstline)
  {
      sm->is_meta = 0;
      sm->num_kernels = firstline;
      sm->kernel_names = (char**)malloc(sm->num_kernels * sizeof(char*));
      sm->kernel_sizes = (int*)calloc(sm->num_kernels, sizeof(int));
    printf("analyzing this number of kernels %d\n", sm->num_kernels);
      char cur_kernel_name[1024]; //if you need more than 1023 characters to name a kernel, you need help
      for (k = 0; k < sm->num_kernels; ++k) {
        assert(!feof(fp));
        fscanf(fp, "%s\n", cur_kernel_name);
        sm->kernel_names[k] = strdup(cur_kernel_name);
        fscanf(fp, "%d\n", &(sm->kernel_sizes[k]));
      }
  }
  else
  {
    sm->is_meta = 1;
    fscanf(fp, "%d\n", &(sm->num_kernels));
    sm->kernel_names = (char**)malloc(sm->num_kernels * sizeof(char*));
    sm->kernel_sizes = (int*)malloc(sm->num_kernels* sizeof(int));
    sm->meta_kernel_sizes = (int*)malloc(sm->num_kernels* sizeof(int));
    sm->meta_w = (double**) malloc(sm->num_kernels*sizeof(double*));
    char cur_kernel_name[1024]; //if you need more than 1023 characters to name a kernel, you need help
    for (k = 0; k < sm->num_kernels; ++k) {
        assert(!feof(fp));
        fscanf(fp, "%s\n", cur_kernel_name);
        printf("Cur kernel name is %s\n", cur_kernel_name);
        sm->kernel_names[k] = strdup(cur_kernel_name);
        assert(!feof(fp));
        fscanf(fp, "%d\n", &(sm->meta_kernel_sizes[k]));
        sm->meta_kernel_sizes[k]++;  //since it also has zero component
        load_meta_kernel(sm, k);
        sm->kernel_sizes[k]=1;
    }
  }
  sm->w_curr.initialize(sm->num_kernels, sm->kernel_sizes, sparm->n_classes, NULL, sparm->do_spm);
  sm->sizePsi = sm->w_curr.total_length-1;
  sm->section_length = sm->w_curr.section_length;
	//sizePsi + 1 is the number of entries in w.  w[0] is 0 ( deliberately) because of indexing issues 
	//with sparse vectors.  w[1] is a bias term - there should be no features that match it.
}

void init_struct_model(char * kernel_info_file, STRUCTMODEL *sm, STRUCT_LEARN_PARM* sparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the dimension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

  read_kernel_info(kernel_info_file, sm, sparm);
}

void box_to_latent_box(Box * box, LATENT_BOX * latent_box) {
	latent_box->position_x_pixel = box->left;
	latent_box->position_y_pixel = box->top;
	latent_box->bbox_width_pixel = box->right - box->left;
	latent_box->bbox_height_pixel = box->bottom - box->top;
}

void init_latent_variables(SAMPLE *sample, IMAGE_KERNEL_CACHE ** cached_images, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
	
  int i;
  /* initialize the RNG */
	init_gen_rand(sparm->rng_seed);
	for (i = 0; i < sample->n; ++i) {
		long init_bbox_index = (long) floor(genrand_res53() * NUM_BBOXES_PER_IMAGE);
		assert(init_bbox_index >= 0);
		assert(init_bbox_index < NUM_BBOXES_PER_IMAGE);
		sample->examples[i].h = make_latent_var(sm);
		int k;
		for (k = 0; k < sm->num_kernels; ++k) {
			box_to_latent_box(&(cached_images[sample->examples[i].x.image_id][0].object_boxes[init_bbox_index]), &(sample->examples[i].h.boxes[k]));
		}
	}

//	for(i=0;i<sample->n;i++)
//	{
//		LATENT_VAR h = make_latent_var(sm);
//		LATENT_BOX random;
//		random.position_x_pixel = sample->examples[i].x.gt_x_pixel;
//		random.position_y_pixel = sample->examples[i].x.gt_y_pixel;
//		random.bbox_width_pixel = sample->examples[i].x.gt_width_pixel; 
//		random.bbox_height_pixel = sample->examples[i].x.gt_height_pixel;
//		//random.position_x_pixel = 1;//(long) floor(genrand_res53()*(sample->examples[i].x.width_pixel-10));
//		//random.position_y_pixel = 1;//(long) floor(genrand_res53()*(sample->examples[i].x.height_pixel-10));
//		//random.bbox_width_pixel = sample->examples[i].x.width_pixel-1;//(long) floor(genrand_res53()*(sample->examples[i].x.width_pixel-random.position_x_pixel-5));
//		//random.bbox_height_pixel = sample->examples[i].x.height_pixel-1;//(long) floor(genrand_res53()*(sample->examples[i].x.height_pixel-random.position_y_pixel-5));
//		
//		for(int j = 0; j < sm->num_kernels; j++)
//		{
//			if (sample->examples[i].y.label == 0) {
//				h.boxes[j].position_x_pixel = 0;
//				h.boxes[j].position_y_pixel = 0;
//				h.boxes[j].bbox_width_pixel = -1;
//				h.boxes[j].bbox_height_pixel = -1;
//			}
//			else {
//				assert(sample->examples[i].y.label==1);
//				h.boxes[j] = random;
//			}
//		}
//		sample->examples[i].h = h;
//	}
//	for (i=0;i<sample->n;i++) {
//		sample->examples[i].h.position_x_pixel = (long) floor(genrand_res53()*(sample->examples[i].x.width_pixel-1));
//		sample->examples[i].h.position_y_pixel = (long) floor(genrand_res53()*(sample->examples[i].x.height_pixel-1));
//		if(sample->examples[i].h.position_x_pixel < 0 || sample->examples[i].h.position_x_pixel >= sample->examples[i].x.width_pixel-1)
//			sample->examples[i].h.position_x_pixel = (long) 0;
//		if(sample->examples[i].h.position_y_pixel < 0 || sample->examples[i].h.position_y_pixel >= sample->examples[i].x.height_pixel-1)
//			sample->examples[i].h.position_y_pixel = (long) 0;
//	}
}

/*keep this around for debugging purposes*/
//int in_bounding_box(int pixel_x, int pixel_y, LATENT_VAR h, STRUCTMODEL * sm) {
//  int bbox_start_x = h.position_x * sm->bbox_step_x;
//  int bbox_start_y = h.position_y * sm->bbox_step_y;
//  int bbox_end_x = bbox_start_x + sm->bbox_width;
//  int bbox_end_y = bbox_start_y + sm->bbox_height;
//  return (pixel_x >= bbox_start_x) && (pixel_y >= bbox_start_y) && (pixel_x < bbox_end_x) && (pixel_y < bbox_end_y);
//}

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

void fill_possible_object_cache(PATTERN x,int kernel_ind, IMAGE_KERNEL_CACHE* ikc, STRUCTMODEL* sm)
{
  char filename[1024];
  sprintf(filename, "%s/%s.txt",BASE_DIR,x.image_path);
  FILE* fp = fopen(filename,"r");
  assert(fp);
  ikc->object_boxes = (Box*) malloc(sizeof(Box)*NUM_BBOXES_PER_IMAGE); 
  for(int p = 0 ; p < NUM_BBOXES_PER_IMAGE ; p++)
  {
    float throwaway_score;
    fscanf(fp, "%d %d %d %d %f", &(ikc->object_boxes[p].left), &(ikc->object_boxes[p].top), &(ikc->object_boxes[p].right),
       &(ikc->object_boxes[p].bottom), &throwaway_score); 
    assert( ikc->object_boxes[p].left);
    assert( ikc->object_boxes[p].top);
    assert( ikc->object_boxes[p].right);
    assert( ikc->object_boxes[p].bottom);
//    printf("New bbox is %d %d %d %d\n", ikc->object_boxes[p].left, ikc->object_boxes[p].top, ikc->object_boxes[p].right, ikc->object_boxes[p].bottom);
  }
  fclose(fp);
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

  /*this will sort points by y, and within that, by x*/
  qsort(ikc->points_and_descriptors, ikc->num_points, sizeof(POINT_AND_DESCRIPTOR), pad_cmp);
  
  /*x.descriptor_top_left_xs[kernel_ind] = ikc->points_and_descriptors[0].x;
  x.descriptor_top_left_ys[kernel_ind] = ikc->points_and_descriptors[0].y;
  
  //Need to cut off last column because of stupid honeycomb nonsense that some idiot from the Netherlands decided to do.
//    cut_off_last_column(ikc);

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
      //printf("pad.x = %d, pad.y = %d\n", ikc->points_and_descriptors[q].x,  ikc->points_and_descriptors[q].y);
    }
  }
  assert((ikc->num_points % p) == 0);
  x.descriptor_num_acrosses[kernel_ind] = ikc->num_points / p;*/
}

void try_cache_image(PATTERN x, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm) {
  int k;
  if (cached_images[x.image_id] == NULL) {
    printf("$"); fflush(stdout);
    cached_images[x.image_id] = (IMAGE_KERNEL_CACHE *)malloc(sm->num_kernels * sizeof(IMAGE_KERNEL_CACHE));
    IMAGE_KERNEL_CACHE * kernel_caches_for_image = cached_images[x.image_id];
    for (k = 0; k < sm->num_kernels; ++k) {
      if(k==0)
      {
        fill_possible_object_cache(x,k, &(kernel_caches_for_image[k]), sm);
      }
      else
      {
        kernel_caches_for_image[k].object_boxes = NULL;
      }
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
int max(int a, int b) {
  if (a > b) return a;
  return b;
}

int min(int a, int b) {
  if (a < b) return a;
  return b;
}

void fill_max_pool(PATTERN x, LATENT_VAR h, int kernel_ind, IMAGE_KERNEL_CACHE ** cached_images, WORD* words, int descriptor_offset, int * num_words, STRUCTMODEL * sm) {
    POINT_AND_DESCRIPTOR * points_and_descriptors = cached_images[x.image_id][kernel_ind].points_and_descriptors;
	int num_descriptors = cached_images[x.image_id][kernel_ind].num_points;
	
    do_max_pooling(points_and_descriptors, h.boxes[kernel_ind], num_descriptors, kernel_ind, words, descriptor_offset, num_words, sm); 
}

int word_cmp(const void * a, const void * b) {
       WORD * word_a = (WORD *)a;
       WORD * word_b = (WORD *)b;
       return word_a->wnum - word_b->wnum;
}

void do_max_pooling(POINT_AND_DESCRIPTOR * points_and_descriptors, LATENT_BOX ourbox, int num_descriptors,int kernel_ind, WORD* words, int descriptor_offset, int * num_words, STRUCTMODEL * sm) {
//	printf("Offset is %d\n", descriptor_offset);
	int init_num_words = *num_words;
	
    int * locations = (int*)calloc(sm->kernel_sizes[kernel_ind], sizeof(int));
	LATENT_BOX h_temp = ourbox;
//	printf("(DMP) bounding box is left %f top  %f width %f, height %f\n", h_temp.position_x_pixel, h_temp.position_y_pixel, h_temp.bbox_width_pixel, h_temp.bbox_height_pixel);
    if(sm->is_meta)
    {
        words[*num_words].weight = 0;
        words[*num_words].wnum= kernel_ind+2;
    }
    int feasible_descriptors = 0;
    for(int i = 0; i< num_descriptors;i++)
	{
		POINT_AND_DESCRIPTOR descriptor = points_and_descriptors[i];
		int position = descriptor.descriptor;
		assert(position+descriptor_offset>1);
		if(!sm->is_meta)
            assert(position+descriptor_offset<sm->sizePsi+1);
		if( (descriptor.x>=ourbox.position_x_pixel) && (descriptor.x<=ourbox.position_x_pixel+ourbox.bbox_width_pixel) &&
				(descriptor.y>=ourbox.position_y_pixel) && (descriptor.y<=ourbox.position_y_pixel+ourbox.bbox_height_pixel) )
		{
			feasible_descriptors++;
            if(!sm->is_meta)
            {
                if (locations[position-1] == 0) {
                    locations[position - 1] = (*num_words);
                    words[*num_words].wnum = position  +descriptor_offset; //position is one indexed, as is descriptor_offset,
                                                                           //so the smallest wnum is 2, which is correct since
                                                                           //the 0th guy is blank and the first guy is 1.
                    assert(words[*num_words].wnum!=0);
                    words[*num_words].weight = (1.0 / W_SCALE);
                    *num_words=*num_words+1;
                }
                else
                {
                    assert(words[locations[position-1]].weight>0);
                    assert(words[locations[position-1]].wnum != 0);
                    //score += sm->w[words[locations[position-1]].wnum];
                    //words[locations[position-1]].weight += 1.0/W_SCALE;
                }
            }
            else
            {
               assert(0); //THIS CODE IS DOING SUM POOLING
               assert(sm->is_meta);
               assert(position>=0);
               assert(position< sm->meta_kernel_sizes[kernel_ind]);
               words[*num_words].weight += (sm->meta_w[kernel_ind][position] / W_SCALE); //THIS IS SUM_POOLING
            }
		}
  }
//  printf("Number feasible descriptors %d %f \n", feasible_descriptors,words[*num_words].weight);
// printf("this guy %d has new weight %f\n", *num_words, words[*num_words].weight);
    if(sm->is_meta)
    {
        *num_words = *num_words + 1; 
    }
//	printf("Score is %f\n", score);
	qsort(&(words[init_num_words]), *num_words - init_num_words, sizeof(WORD), word_cmp);
	free(locations);
}

/*void zero_svector_parts(int * valid_kernels, SVECTOR * fvec, STRUCTMODEL * sm) {
	int subset;
	for(subset = 0; subset< NUM_WINDOWS;subset++)
	{
		int word_ind = 0;
		int current_kernel_ind = 0;
		int kernel_start = subset*sm->sizeSinglePsi;
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
}*/
int binsearch(POINT_AND_DESCRIPTOR* points_and_descriptors, int length, double target)
{
    int lower_bound = 0;
    int upper_bound = length-1; //inclusive

    while(lower_bound != upper_bound)
    {
        int new_val = ((lower_bound + upper_bound)/2);
        if(points_and_descriptors[new_val].y >= target)
        {
            upper_bound = new_val;
        }
        else
        {
            lower_bound = new_val+1;
        }
    }
    return lower_bound;
}
void single_psi(PATTERN x, LABEL y, LATENT_VAR h, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,int box_num, bool in_bb, kernel_obj& results) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/

  try_cache_image(x, cached_images, sm);
  
  for(int kernel_ind = 0 ; kernel_ind < sm->num_kernels;  kernel_ind++)
  {
      POINT_AND_DESCRIPTOR * points_and_descriptors = cached_images[x.image_id][kernel_ind].points_and_descriptors;
      int num_descriptors = cached_images[x.image_id][kernel_ind].num_points;
      int startpoint = binsearch(points_and_descriptors, num_descriptors,h.boxes[kernel_ind].position_y_pixel);
      for(int index = startpoint ; index < num_descriptors  ; index++)
      {
        if( (points_and_descriptors[index].x >= h.boxes[kernel_ind].position_x_pixel) &&
            (points_and_descriptors[index].x <= h.boxes[kernel_ind].position_x_pixel+h.boxes[kernel_ind].bbox_width_pixel) )
        {
            if(points_and_descriptors[index].y > h.boxes[kernel_ind].position_y_pixel+h.boxes[kernel_ind].bbox_height_pixel)
                break;
//            assert(points_and_descriptors[index].y >= h.boxes[kernel_ind].position_y_pixel);
//            assert(points_and_descriptors[index].y <= h.boxes[kernel_ind].position_y_pixel+h.boxes[kernel_ind].bbox_height_pixel);
            results.set(kernel_ind, points_and_descriptors[index].descriptor-1, box_num, in_bb, y.label, 1/W_SCALE );
        }
      }
  }
}

LATENT_VAR choose_subset(LATENT_VAR h, int subset,  STRUCT_LEARN_PARM *sparm, STRUCTMODEL* sm)
{
	LATENT_VAR h_out = make_latent_var(sm);
	for(int i = 0 ; i < sm->num_kernels; i++)
	{
		if( (!sparm->do_spm))
		{
			if(subset == 0)
			{
				h_out.boxes[i]=h.boxes[i];
				continue;
			}
			else 
			{
			   assert(0);	
 			}
		}
		//DOING SPM
		h_out.boxes[i] = h.boxes[i];

		if(subset == 0)
		{
			continue;
		}
		else
		{
			h_out.boxes[i].bbox_width_pixel = h_out.boxes[i].bbox_width_pixel/2;
			h_out.boxes[i].bbox_height_pixel = h_out.boxes[i].bbox_height_pixel/2;
			
			if(subset==1)
			{
				h_out.boxes[i].position_x_pixel = h_out.boxes[i].position_x_pixel; 
				h_out.boxes[i].position_y_pixel = h_out.boxes[i].position_y_pixel;
			}
			else if(subset==2)
			{
				h_out.boxes[i].position_x_pixel = h_out.boxes[i].position_x_pixel+h_out.boxes[i].bbox_width_pixel;
				h_out.boxes[i].position_y_pixel = h_out.boxes[i].position_y_pixel;
			}
			else if(subset==3)
			{
				h_out.boxes[i].position_x_pixel = h_out.boxes[i].position_x_pixel;
				h_out.boxes[i].position_y_pixel = h_out.boxes[i].position_y_pixel+h_out.boxes[i].bbox_height_pixel;
			}
			else
			{
				assert(subset==4);
				h_out.boxes[i].position_x_pixel = h_out.boxes[i].position_x_pixel+h_out.boxes[i].bbox_width_pixel;
				h_out.boxes[i].position_y_pixel = h_out.boxes[i].position_y_pixel+h_out.boxes[i].bbox_height_pixel;
			}
		}
	}
	return h_out;
}
LATENT_VAR make_lv_wholeimage(PATTERN x, STRUCTMODEL* sm)
{
    LATENT_VAR out = make_latent_var(sm);
    LATENT_BOX whole_image;
    whole_image.position_x_pixel = 0;
    whole_image.position_y_pixel = 0;
    whole_image.bbox_width_pixel = x.width_pixel;
    whole_image.bbox_height_pixel = x.height_pixel;
    for(int k = 0 ; k < sm->num_kernels ; k++)
        out.boxes[k] = whole_image;
    return out;
}

double time_delta_micros(struct timeval& start_time, struct timeval& end_time)
{
    return 1e6 * (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec);
}



SVECTOR* psi_helper(PATTERN x, LABEL y, LATENT_VAR h, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, bool do_whole_image) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/

  try_cache_image(x, cached_images, sm);
  SVECTOR* fvec;
  //binary labelling for now - 1 means there's a car, 0 means there's no car
		//struct timeval start_time;
		//struct timeval end_time;
//		gettimeofday(&start_time, NULL);
        LATENT_VAR whole_image = make_lv_wholeimage(x,sm);
        kernel_obj result;
        result.initialize(sm->num_kernels, sm->kernel_sizes, sparm->n_classes, NULL, sparm->do_spm);
		for(int subset = 0; ((subset<5) && sparm->do_spm) || (subset<1); subset++)
		{
			LATENT_VAR subset_box = choose_subset(h,subset,sparm,sm);
			single_psi(x,y,subset_box,cached_images,valid_kernels,sm,sparm,subset,true,result);
			free_latent_var(subset_box);
       		     	if(do_whole_image)
            		{
                		LATENT_VAR subset_box_whole_image = choose_subset(whole_image,subset,sparm,sm);
                		single_psi(x,y,subset_box_whole_image,cached_images,valid_kernels,sm,sparm,subset,false,result);
                		free_latent_var(subset_box_whole_image);
            		}
		}
        fvec = result.get_svec(y.label);
        result.cleanup();
        free_latent_var(whole_image);
		//gettimeofday(&end_time, NULL);
        //printf("Psi took %f micros\n", time_delta_micros(start_time, end_time));
        return fvec;
}

double compute_w_T_psi_helper(PATTERN *x, LATENT_VAR h, int classi, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, bool do_whole_image) {
  double w_T_psi;
  LABEL y;
  y.label = classi;
  SVECTOR * psi_vect = psi_helper(*x, y, h, cached_images, valid_kernels, sm, sparm,do_whole_image);
	int curr = 0;
	int max_so_far = -1;
	while(psi_vect->words[curr].wnum !=0)
	{
		if(psi_vect->words[curr].wnum<max_so_far)
			assert(0);
		else
			max_so_far = psi_vect->words[curr].wnum;
		curr++;
	}
  w_T_psi = sprod_ns(sm->w_curr.get_vec(), psi_vect);
  free_svector(psi_vect);
  return w_T_psi;
}

double compute_w_T_psi(PATTERN *x, LATENT_VAR h, int classi, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
    return compute_w_T_psi_helper(x, h, classi, cached_images, valid_kernels, sm, sparm, true);
}

SVECTOR* psi(PATTERN x, LABEL y, LATENT_VAR h, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
    return psi_helper(x, y, h, cached_images, valid_kernels, sm, sparm, true);
}

void try_new_latent(PATTERN x,LABEL y,IMAGE_KERNEL_CACHE ** cached_images,int* valid_kernels,STRUCTMODEL* sm,STRUCT_LEARN_PARM* sparm,double* max_score,LATENT_VAR* h_best,LABEL* y_best,LABEL y_curr,LATENT_VAR h_curr, double loss)
{
	double score = loss + compute_w_T_psi(&x,h_curr,y_curr.label,cached_images,valid_kernels,sm, sparm);
	if(score > *max_score)
	{
		*max_score=score;
		*h_best = h_curr;
		*y_best = y_curr;
	}	
}

LATENT_VAR make_latent_var(STRUCTMODEL* sm)
{
	LATENT_VAR var;
	var.boxes = (LATENT_BOX*)malloc(sm->num_kernels*sizeof(LATENT_BOX));
	return var;
}

void compute_highest_scoring_latents_hallucinate(PATTERN x,LABEL y,IMAGE_KERNEL_CACHE ** cached_images,int* valid_kernels,STRUCTMODEL* sm,STRUCT_LEARN_PARM* sparm,double* max_score,LATENT_VAR* h_best,LABEL* y_best,LABEL y_curr)
{
  assert(0);
/*  double loss = (y_curr.label == y.label) ? 0 : 1;

	if(y_curr.label==0)
	{
		if(loss>*max_score)
		{
			*max_score = loss;
			*y_best = y_curr;
			for(int i = 0 ; i < sm->num_kernels; i++)
			{
				h_best->boxes[i].position_x_pixel=0;
				h_best->boxes[i].position_y_pixel=0;
				h_best->boxes[i].bbox_width_pixel=-1;
				h_best->boxes[i].bbox_height_pixel=-1;
			}
		}
	}
	else
	{
		int offset = 1;
		LATENT_VAR h_latent_var =  make_latent_var(sm);
		struct timeval start_time;
		gettimeofday(&start_time, NULL);
		int* single_valid_kernels = (int*)calloc(sm->num_kernels, sizeof(int));
		for(int k = 0  ; k < sm->num_kernels; k++)
		{
			single_valid_kernels[k] = 1;
			assert(y_curr.label==1);

		 	int total_indices = cached_images[x.example_id][k].num_points;

			double* argxpos = (double*)malloc(sizeof(double)*total_indices);
			double* argypos = (double*)malloc(sizeof(double)*total_indices);
			double* argclst = (double*)malloc(sizeof(double)*total_indices);
            
            int size_w = sm->is_meta ? (sm->sizeSingleMetaPsi-sm->num_kernels) : (sm->sizeSinglePsi-1);
            double* w = (double*) malloc(sizeof(double)*size_w);

            if(!sm->is_meta)
            {
                memcpy(w, &sm->w[2], sizeof(double)*(size_w));
            }
            else
            {
               int curr_index = 0;
               for(int i = 0 ; i < sm->num_kernels; i++)
               {
                   // printf("curr index is %d and amount to write is %d with limit %d\n", curr_index, sm->meta_kernel_sizes[i],size_w);
                    for(int j = 1 ; j< sm->meta_kernel_sizes[i] ; j++)
                    {
                        assert((curr_index+j-1)<size_w);
                        assert(curr_index+j-1>=0);
                        w[curr_index+j-1] = (sm->meta_w[i][j]/W_SCALE)*sm->w[i+2];
                        //printf("%d ", curr_index+j);
                    }
                    curr_index+= sm->meta_kernel_sizes[i]-1;
               }
            }
			int factor = 20;
			int curr_point = 0;
			for(int i = 0 ; i<cached_images[x.example_id][k].num_points;i++)
			{
				argxpos[curr_point] = (cached_images[x.example_id][k].points_and_descriptors[i].x);
				argypos[curr_point] = (cached_images[x.example_id][k].points_and_descriptors[i].y);
				argclst[curr_point] = cached_images[x.example_id][k].points_and_descriptors[i].descriptor+offset-2;
				assert(argclst[curr_point]>=0);
				assert(argclst[curr_point]<size_w);
				curr_point++;
			}
			offset += sm->kernel_sizes[k];
			int solvedExactly=1;
			assert(curr_point == total_indices);
			int N = sparm->do_spm ? 2 : 1;
			Box ourbox = pyramid_search(total_indices, 1+(int)(x.width_pixel), 1+(int)(x.height_pixel),
											 argxpos, argypos, argclst,
												size_w, N, w,
												1e9, solvedExactly, factor,
                                                NUM_BBOXES_PER_IMAGE, cached_images[x.example_id][0].object_boxes ); 

			LATENT_BOX h_temp;
			h_temp.position_x_pixel=ourbox.left; 
			h_temp.position_y_pixel=ourbox.top;
			h_temp.bbox_width_pixel=(ourbox.right-ourbox.left);
			h_temp.bbox_height_pixel=(ourbox.bottom-ourbox.top);

			h_latent_var.boxes[k] = h_temp;
			single_valid_kernels[k] = 0;
			free(argxpos);
			free(argypos);
			free(argclst);
            free(w);
		}
		free(single_valid_kernels);
		double ourscore = compute_w_T_psi(&x, h_latent_var, y_curr.label,cached_images, valid_kernels, sm, sparm);
//		printf("our total score is %f\n", ourscore);
		if(ourscore+loss>*max_score)
		{
			*max_score = ourscore+loss;
			*y_best = y_curr;
			free_latent_var(*h_best);
			*h_best=h_latent_var;
		}
		else
		{
			free_latent_var(h_latent_var);
		}
		struct timeval end_time;
		gettimeofday(&end_time, NULL);
		//double microseconds = 1e6 * (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec);
//		printf("ESS (hallucinating) took %f \n", microseconds/1000);
	}*/
}

void compute_highest_scoring_latents(PATTERN x,LABEL y,IMAGE_KERNEL_CACHE ** cached_images,int* valid_kernels,STRUCTMODEL* sm,STRUCT_LEARN_PARM* sparm,double* max_score,LATENT_VAR* h_best,LABEL* y_best,LABEL y_curr, bool during_learning)
{
	if (during_learning) assert(!x.also_correct[y_curr.label]); //either y_curr is the "official" correct class or it's incorrect
	double loss = (y_curr.label == y.label) ? 0 : 1;
	if (!during_learning) assert(loss == 0);
        LATENT_VAR this_best =  make_latent_var(sm);
        double this_best_score = -DBL_MAX;
        for(int i = 0 ; i < NUM_BBOXES_PER_IMAGE; i ++)
        {
            Box box = cached_images[x.image_id][0].object_boxes[i];
            LATENT_BOX h_temp_box;
	    box_to_latent_box(&box, &h_temp_box);
            LATENT_VAR h_temp = make_latent_var(sm);
            for(int j = 0  ; j < sm->num_kernels ; j++)
                h_temp.boxes[j] = h_temp_box;

    		double ourscore = compute_w_T_psi_helper(&x, h_temp, y_curr.label,cached_images, valid_kernels, sm, sparm,false);
            if(ourscore>this_best_score)
            {
                this_best_score = ourscore;
                free_latent_var(this_best);
                this_best = h_temp;
            }
            else
                free_latent_var(h_temp);
        }
        double ourscore = compute_w_T_psi_helper(&x, this_best, y_curr.label,cached_images, valid_kernels, sm, sparm,true);
        if(ourscore+loss>*max_score)
        {
            *max_score = ourscore+loss;
            *y_best = y_curr;
            free_latent_var(*h_best);
            *h_best = this_best;
        }
        else
        {
            free_latent_var(this_best);
        }
}

double get_classifier_score(PATTERN x, LABEL y_guess, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm, STRUCT_LEARN_PARM * sparm) {
	double max_score = -DBL_MAX;
	LATENT_VAR h;
	LABEL y;
	compute_highest_scoring_latents(x, y_guess, cached_images, NULL, sm, sparm, &max_score, &h, &y, y_guess, false);
	return max_score;
}

double classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int impute) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
  
//  printf("sparm->n_classes = %d, x.width = %d, x.height = %d\n", sparm->n_classes, x.width, x.height);
	int cur_class;
	double max_score;
	max_score = -DBL_MAX;
	LABEL y_curr;

	for(cur_class = 0; cur_class<sparm->n_classes; cur_class++)
	{
    	   	y_curr.label = cur_class;
		compute_highest_scoring_latents(x,y_curr,cached_images,NULL,sm,sparm,&max_score,h,y,y_curr,false);
	}
	return max_score;
}

void initialize_most_violated_constraint_search(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, double * max_score, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	copy_latent_var(hstar, hbar, sm);
  ybar->label = y.label;
  *max_score = compute_w_T_psi(&x, *hbar, ybar->label, cached_images, valid_kernels, sm, sparm);
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

	int cur_class;
	double max_score;
	
	//make explicit the idea that (y, hstar) is what's returned if the constraint is not violated
	initialize_most_violated_constraint_search(x, hstar, y, ybar, hbar, &max_score, cached_images, valid_kernels, sm, sparm);

	int isValid = 0;
	for(int i = 0 ; i < sm->num_kernels;i++)
	{
		if(valid_kernels[i])
			isValid = 1;
	}

	if(isValid)
	{
		for(cur_class = 0; cur_class<sparm->n_classes;cur_class++)
		{
			if (!x.also_correct[cur_class]) {
				LABEL y_curr;
				y_curr.label = cur_class;
				if(cur_class != y.label) //so we take a penalty of one for the misclassification.
				{
					if(sparm->do_hallucinate)
						compute_highest_scoring_latents_hallucinate(x,y,cached_images,valid_kernels,sm,sparm,&max_score,hbar,ybar,y_curr);
					else
						compute_highest_scoring_latents(x,y,cached_images,valid_kernels,sm,sparm,&max_score,hbar,ybar,y_curr,true);
				}
			}
		}
	}
	return;
}

void find_most_violated_constraint_differenty(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	assert(0);
  return;
}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  //printf("width = %d, height = %d\n", x.width, x.height);
  //time_t start_time = time(NULL);
  LATENT_VAR h = make_latent_var(sm);
	/*else {
		assert(y.label==1);
		h.position_x_pixel = x.gt_x_pixel;
		h.position_y_pixel = x.gt_y_pixel;
		h.bbox_width_pixel = x.gt_width_pixel; 
		h.bbox_height_pixel = x.gt_height_pixel;

    return h;
	}*/
	double MAX_SCORE_ATTAINED = -DBL_MAX;
	LABEL garbage; //will get set to whatever the variable y holds.
	garbage.label=0;
	compute_highest_scoring_latents(x,y,cached_images,NULL,sm,sparm,&MAX_SCORE_ATTAINED,&h,&garbage, y, true);

	assert(garbage.label==y.label);

  return(h); 
}


void print_lv(LATENT_VAR h)
{
	printf("%f %f %f %f\n", h.boxes[0].position_y_pixel,  h.boxes[0].position_x_pixel,  h.boxes[0].bbox_height_pixel,  h.boxes[0].bbox_width_pixel);
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
  
  for (i=1;i<sm->sizePsi+1;i++) {
    fprintf(modelfl, "%d:%.16g\n", i, sm->w_curr.get_vec()[i]);
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
  
  double* w = sm->w_curr.get_vec();
  while (!feof(modelfl)) {
    fscanf(modelfl, "%d:%lf", &fnum, &fweight);
		w[fnum] = fweight;
  }

  fclose(modelfl);
}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/
  int k;
  

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
	free(h.boxes);
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
  sparm->n_classes = -1;
  sparm->pos_neg_cost_ratio = 1.0;
  sparm->C = 10000;
  sparm->prox_weight  = 0 ;
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
    case 'j' : i++; sparm->pos_neg_cost_ratio = atof(sparm->custom_argv[i]); break;
    case 'c' : i++; sparm->C = atof(sparm->custom_argv[i]); break;
    case 's': i++; sparm->rng_seed = atoi(sparm->custom_argv[i]); break;
    case 'n': i++; sparm->n_classes = atoi(sparm->custom_argv[i]); break;
    case 't': i++; sparm->margin_type = atoi(sparm->custom_argv[i]); break;
    case 'l': i++; sparm->do_spm = atoi(sparm->custom_argv[i]); break;
    case 'h': i++; sparm->do_hallucinate = atoi(sparm->custom_argv[i]); break;
    case 'p': i++; sparm->prox_weight = atoi(sparm->custom_argv[i]); break;
    default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
	assert(sparm->n_classes > 0);
}

void copy_label(LABEL l1, LABEL *l2)
{
	l2->label = l1.label;
}

void copy_latent_var(LATENT_VAR lv1, LATENT_VAR *lv2,STRUCTMODEL* sm)
{
	memcpy(lv2->boxes,  lv1.boxes, sizeof(LATENT_BOX)*sm->num_kernels);
}

void print_latent_var(PATTERN x, LATENT_VAR h, FILE *flatent)
{
  char img_num_str[1024];
  char * img_num_ptr = img_num_str;
  strcpy(img_num_ptr, x.image_path);
//  img_num_ptr = strchr(img_num_ptr, (int)('/'));
//  img_num_ptr++;
//  img_num_ptr = strchr(img_num_ptr, (int)('/'));
//  img_num_ptr++;
	if(flatent)
	{
  	fprintf(flatent,"%s %f %f %f %f ", img_num_ptr, h.boxes[0].position_x_pixel,h.boxes[0].position_y_pixel,h.boxes[0].bbox_width_pixel, h.boxes[0].bbox_height_pixel);
		fflush(flatent);
	}
	else
	{
  	printf("%s %f %f %f %f ", img_num_ptr, h.boxes[0].position_x_pixel,h.boxes[0].position_y_pixel,h.boxes[0].bbox_width_pixel, h.boxes[0].bbox_height_pixel);
		fflush(flatent);
	}
}
void read_latent_var(LATENT_VAR *h, FILE *finlatent)
{
    fscanf(finlatent,"%lf%lf",&h->boxes[0].position_x_pixel,&h->boxes[0].position_y_pixel);
}

void print_label(LABEL l, FILE	*flabel)
{
	fprintf(flabel,"%d ",l.label);
	fflush(flabel);
}
