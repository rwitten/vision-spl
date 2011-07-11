/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_classify.c                                       */
/*                                                                      */
/*   Classification Code for Latent SVM^struct                          */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 9.Nov.08                                                     */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/
#include <math.h>
#include <stdio.h>
#include "svm_struct_latent_api.h"
#include "./svm_light/svm_learn.h"

#define KERNEL_INFO_FILE "data/kernel_info.txt"
#define max(x,y) ( ((x)>(y)) ? (x) : (y))
#define C 100000
#define J 1
void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, char *labelfile, char *latentfile, char *inlatentfile, char *resultfile, STRUCT_LEARN_PARM *sparm);

double regularizaton_cost(double* w, long num_entries)
{
    long k;
    double cost = 0;
    for(k=0; k<=num_entries;k++) {
        cost += w[k]*w[k]*.5;
    }
    return cost;
}

int main(int argc, char* argv[]) {
  double avghingeloss,avgloss,l,hinge_l;
  LABEL y;
  long i, correct;

  char testfile[1024];
  char modelfile[1024];
	char labelfile[1024];
	char latentfile[1024];
	char resultfile[1024];
    char inlatentfile[1024];
	FILE	*flabel;
	FILE	*flatent;
    FILE    *finlatent;

  STRUCTMODEL model;
  STRUCT_LEARN_PARM sparm;
  LEARN_PARM lparm;
  KERNEL_PARM kparm;

  SAMPLE testsample;
  
  LATENT_VAR h;

  /* read input parameters */
  read_input_parameters(argc,argv,testfile,modelfile,labelfile,latentfile,inlatentfile,resultfile,&sparm);
	flabel = fopen(labelfile,"w");
	flatent = fopen(latentfile,"w");
    if(inlatentfile[0]!='\0')
        finlatent = fopen(inlatentfile,"r"); 
    else
        finlatent = NULL;

  init_struct_model(get_sample_size(testfile), KERNEL_INFO_FILE, &model);

  read_struct_model(modelfile, &model);

  IMAGE_KERNEL_CACHE ** cached_images = init_cached_images(&model);

  /* read test examples */
	printf("Reading test examples..."); fflush(stdout);
	testsample = read_struct_examples(testfile, &model, &sparm);
	printf("done.\n");

  
  avghingeloss = 0.0;
  correct = 0;
  int impute = (int) (finlatent == NULL);
  int *valid_example_kernel = (int *) malloc(5*sizeof(int));
  for(i = 0; i < 5; i++)
    valid_example_kernel[i] = 1;
  printf("IS PLANNING TO IMPUTE %d\n",impute);
  for (i=0;i<testsample.n;i++) {
    if(finlatent) {
        read_latent_var(&h,finlatent);
        //printf("%d %d\n",h.position_x,h.position_y);
    }
    double score = classify_struct_example(testsample.examples[i].x,&y,&h,cached_images,&model,&sparm,impute);
    l = loss(testsample.examples[i].y,y,h,&sparm);
    if (l<.1) correct++;

		print_label(y,flabel);
		fprintf(flabel,"\n"); fflush(flabel);

		print_latent_var(h,flatent);
		fprintf(flatent,"\n"); fflush(flatent);
      LABEL ybar;
      LATENT_VAR hbar;
      SVECTOR *fy, *fybar;
      testsample.examples[i].h.position_x = h.position_x;
      testsample.examples[i].h.position_y = h.position_y;
      find_most_violated_constraint_marginrescaling(testsample.examples[i].x,testsample.examples[i].h,testsample.examples[i].y,&ybar,&hbar,cached_images,valid_example_kernel,&model,&sparm);
      fy = psi(testsample.examples[i].x,testsample.examples[i].y,testsample.examples[i].h,cached_images,valid_example_kernel,&model,&sparm);
      fybar = psi(testsample.examples[i].x,ybar,hbar,cached_images,valid_example_kernel,&model,&sparm);
    double lossval = loss(testsample.examples[i].y,ybar,hbar,&sparm);
    hinge_l = (lossval+sprod_ns(model.w,fybar)-sprod_ns(model.w,fy));

    /*double* total = malloc(sizeof(double) * (model.sizePsi+1));
    int iter;
    for(iter= 0 ; iter < model.sizePsi+1; iter++)
        total[iter] = 1;
    double l2norm = sqrt(sprod_ss(fy,fy));
    double l1norm = sprod_ns(total, fy);
    printf("feature l1norm %f l2norm %f\n", l1norm, l2norm);
    l2norm = sqrt(sprod_ss(fybar,fybar));
    l1norm = sprod_ns(total, fy);
    free(total);
    printf("feature other choice l1norm %f l2norm %f\n", l1norm, l2norm);
    printf("hinge loss %f\n", hinge_l);*/

    if(testsample.examples[i].y.label)
        hinge_l *= J;
    avghingeloss += hinge_l;

    free_label(y);
    free_latent_var(h); 
  }
	fclose(flabel);
	fclose(flatent);
    if(finlatent)
        fclose(finlatent);

  double w_cost = regularizaton_cost(model.w, model.sizePsi);
  avghingeloss =  avghingeloss/testsample.n;
  printf("\n");
  printf("Objective Value %d %f\n\n\n", C, (C * avghingeloss) + w_cost);
  printf("Average hinge loss on dataset: %.4f\n", avghingeloss);
  printf("Zero/one error on test set: %.4f\n", 1.0 - ((float) correct)/testsample.n);

  FILE *fresult = fopen(resultfile,"w");
  fprintf(fresult,"%.4f\n",1.0 - ((float) correct)/testsample.n);
  fclose(fresult);

  free_cached_images(cached_images, &model);
  free_struct_sample(testsample);
  free_struct_model(model,&sparm);

  return(0);

}


void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, char *labelfile, char *latentfile, char *inlatentfile, char *resultfile, STRUCT_LEARN_PARM *sparm) {

  long i;
  
  /* set default */
  strcpy(modelfile, "lssvm_model");
  strcpy(labelfile, "lssvm_label");
  strcpy(latentfile, "lssvm_latent");
  strcpy(resultfile, "lssvm_result");
  strcpy(inlatentfile,"lssvm_inlatent");
  sparm->custom_argc = 0;

  for (i=1;(i<argc)&&((argv[i])[0]=='-');i++) {
    switch ((argv[i])[1]) {
      case '-': strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);i++; strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);break;  
      default: printf("\nUnrecognized option %s!\n\n",argv[i]); exit(0);    
    }
  }

  if (i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    exit(0);
  }

  strcpy(testfile, argv[i]);
	if(i+1<argc)
  	strcpy(modelfile, argv[i+1]);
	if(i+2<argc)
		strcpy(labelfile,argv[i+2]);
	if(i+3<argc)
		strcpy(latentfile,argv[i+3]);
        if(i+4<argc)
	        strcpy(resultfile,argv[i+4]);
    if(i+5<argc)
        strcpy(inlatentfile,argv[i+5]);
    else
        inlatentfile[0] = '\0';

  parse_struct_parameters(sparm);

}
