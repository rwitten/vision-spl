#ifndef SVM_STRUCT_LATENT_API_TYPES_H
#define SVM_STRUCT_LATENT_API_TYPES_H

/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api_types.h                                      */
/*                                                                      */
/*   API type definitions for Latent SVM^struct                         */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 30.Sep.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

# include "svm_light/svm_common.h"
#include <pthread.h>
#include "ess.hh"
#include "kernel.hh"

typedef struct pattern {
  /*
    Type definition for input pattern x
  */
  int example_id;
  char image_path[1000];
  int image_id; //makes it easier to avoid looking at duplicates
  double example_cost;
  int width_pixel;
  int height_pixel;
  int * descriptor_num_downs;
  int * descriptor_num_acrosses;
  int * descriptor_top_left_xs;
  int * descriptor_top_left_ys;
	//int gt_width_pixel;
	//int gt_height_pixel;
	//int gt_x_pixel;
	//int gt_y_pixel;
	int * also_correct; //contains 1 and indices of classes OTHER THAN THE "OFFICIAL" CORRECT CLASS that are also correct (marginrescaling won't work if we ignore the "official" correct class)
} PATTERN;

typedef struct point_and_descriptor {
  int x;
  int y;
  int descriptor;
} POINT_AND_DESCRIPTOR;

typedef struct image_kernel_cache {
  int num_points;
  POINT_AND_DESCRIPTOR * points_and_descriptors;
  Box* object_boxes;
} IMAGE_KERNEL_CACHE;

typedef struct label {
  /*
    Type definition for output label y
  */
  int label; /* {0,1} */
} LABEL;

typedef struct _sortStruct {
	  double val;
	  int index;
}  sortStruct;

typedef struct latent_box {
  /*
    Type definition for latent variable h
  */
  double position_x_pixel; /* starting position of object */
	double position_y_pixel;
	double bbox_width_pixel;
	double bbox_height_pixel;
} LATENT_BOX;

typedef struct latent_var {
	LATENT_BOX* boxes;
} LATENT_VAR;

typedef struct example {
  PATTERN x;
  LABEL y;
  LATENT_VAR h;
} EXAMPLE;

typedef struct sample {
  int n;
  EXAMPLE *examples;
} SAMPLE;


typedef struct structmodel {
  //double *w;          /* pointer to the learned weights */
  kernel_obj w_curr;
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* sizePsi+1 is length of w. */
  long section_length;
  char kernel_info_file[1024]; //where the config is at
 
  int is_meta;
  double** meta_w;
  int* meta_kernel_sizes;
  int sizeSingleMetaPsi;
  int sizeMetaPsi;
  char filestub[1024];
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
  long n;             /* number of examples */
  int num_kernels;
  int * kernel_sizes;
  char ** kernel_names;
  long num_distinct_images;
} STRUCTMODEL;


typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve
				  quadratic program */
  long newconstretrain;        /* number of new constraints to
				  accumulate before recomputing the QP
				  solution */
  double C;                    /* trade-off between margin and loss */
  double pos_neg_cost_ratio;
  char   custom_argv[20][1000]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm, 
				  2 -> L2-norm */
  int    loss_type;            /* selected loss function from -r
				  command line option. Select between
				  slack rescaling (1) and margin
				  rescaling (2) */
  int    loss_function;        /* select between different loss
				  functions via -l command line
				  option */
  /* add your own variables */
	double init_valid_fraction;
	int optimizer_type;
        int margin_type;
	int rng_seed;
	int size_hog;
	int n_classes;
  int multi_kernel_spl;
	int do_spm;
	int do_hallucinate;
	double prox_weight;
} STRUCT_LEARN_PARM;

typedef struct spl_variable_struct {
  int num_valid_examples;
  int num_valid_kernels;
  int * valid_examples; //boolean
  int * valid_kernel_indices;
  int ** valid_kernel_indices_per_example; //has an array for each example (including those with no valid kernels)
} SPL_VAR_STRUCT;

typedef struct psi_job {
    int m;
    int* curr_task;
    int* completed_tasks;
    pthread_mutex_t* curr_lock;
    pthread_mutex_t* completed_lock;
    EXAMPLE* ex_list;
    LABEL* ybar_list;
    LATENT_VAR* hbar_list;
    IMAGE_KERNEL_CACHE** cached_images;
    STRUCTMODEL* sm;
    STRUCT_LEARN_PARM* sparm;
    SVECTOR** output_vectors;
    int** valid_example_kernels;
} psi_job;

typedef struct fmvc_job {
    int m;
    int* curr_task;
    int* completed_tasks;
    pthread_mutex_t* curr_lock;
    pthread_mutex_t* completed_lock;
    EXAMPLE* ex_list;
    LABEL* ybar_list;
    LATENT_VAR* hbar_list;
    IMAGE_KERNEL_CACHE** cached_images;
    STRUCTMODEL* sm;
    STRUCT_LEARN_PARM* sparm;
    int* valid_examples;
    int** valid_example_kernels;
} fmvc_job;

#endif
