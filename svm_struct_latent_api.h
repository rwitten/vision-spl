#ifndef SVM_STRUCT_LATENT_API_H
#define SVM_STRUCT_LATENT_API_H

#include "./svm_light/svm_common.h"
#include "svm_struct_latent_api_types.h"
#include <float.h>

void cut_off_last_column(IMAGE_KERNEL_CACHE * ikc);
int pad_cmp(const void * a, const void * b);
int get_sample_size(char * file);
IMAGE_KERNEL_CACHE ** init_cached_images(STRUCTMODEL * sm);
void free_cached_images(IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm);
SAMPLE read_struct_examples(char *file, STRUCTMODEL * sm, STRUCT_LEARN_PARM *sparm);
int get_num_bbox_positions(int image_length, int bbox_length, int bbox_step_length);
void read_kernel_info(char * kernel_info_file, STRUCTMODEL * sm);
void init_struct_model(int sample_size, char * kernel_info_file, STRUCTMODEL *sm);
void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
int in_bounding_box(int pixel_x, int pixel_y, LATENT_VAR h, STRUCTMODEL * sm);
int bbox_coord_to_pixel_coord(int bbox_coord, int bbox_step);
int pixel_coord_to_descriptor_coord(int pixel_coord, int descriptor_tl_offset, int descriptor_spacing);
FILE * open_kernelized_image_file(PATTERN x, int kernel_ind, STRUCTMODEL * sm);
void fill_image_kernel_cache(PATTERN x, int kernel_ind, IMAGE_KERNEL_CACHE * ikc, STRUCTMODEL * sm);
void try_cache_image(PATTERN x, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm);
SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, IMAGE_KERNEL_CACHE ** cached_images, int* valid_kernels,STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
double compute_w_T_psi(PATTERN *x, int position_x, int position_y, int class, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
double classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int impute);
void initialize_most_violated_constraint_search(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, double * max_score, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void find_most_violated_constraint_marginrescaling(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, IMAGE_KERNEL_CACHE ** cached_images, int* valid_kernels,STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void find_most_violated_constraint_differenty(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, IMAGE_KERNEL_CACHE ** cached_images, int* valid_kernels,STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void zero_svector_parts(int * valid_kernels, SVECTOR * fvec, STRUCTMODEL * sm);
double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm);
void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void read_struct_model(char *model_file, STRUCTMODEL * sm);
void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm);
void free_pattern(PATTERN x);
void free_label(LABEL y);
void free_latent_var(LATENT_VAR h);
void free_struct_sample(SAMPLE s);
void parse_struct_parameters(STRUCT_LEARN_PARM *sparm);
void copy_label(LABEL l1, LABEL *l2);
void copy_latent_var(LATENT_VAR lv1, LATENT_VAR *lv2);
void print_latent_var(LATENT_VAR h, FILE *flatent);
void read_latent_var(LATENT_VAR *h, FILE *finlatent);
void print_label(LABEL l, FILE	*flabel);

void fill_max_pool(PATTERN x, LATENT_VAR h, int kernel_ind, IMAGE_KERNEL_CACHE ** cached_images, WORD * words, int descriptor_offset, int * num_words, STRUCTMODEL * sm);

void do_max_pooling(POINT_AND_DESCRIPTOR * points_and_descriptors, int start_x, int start_y, int num_across, int num_down, int total_num_down, int kernel_ind, WORD * words, int descriptor_offset, int * num_words, STRUCTMODEL * sm);

#endif
