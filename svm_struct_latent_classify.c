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
#include <assert.h>
#include "svm_struct_latent_api.h"
#include "./svm_light/svm_learn.h"

#define KERNEL_INFO_FILE "data/kernel_info.txt"
#define max(x,y) ( ((x)>(y)) ? (x) : (y))


void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, char *labelfile, char *latentfile, char *inlatentfile, char *scorefile, STRUCT_LEARN_PARM *sparm);
double get_hinge_l_from_pos_score(double pos_score, LABEL gt)
{
	return max(1 - 2*((double)gt.label-.5)*pos_score,0);
}

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
  double avghingeloss,avgloss,l;
  LABEL y;
  long i, correct;
  int log_psis;
  double weighted_correct;

  char testfile[1024];
  char modelfile[1024];
	char labelfile[1024];
	char latentfile[1024];
	char scorefile[1024];
    char inlatentfile[1024];
    char psiposfile[1024];
    char psinegfile[1024];
	FILE	*flabel;
	FILE	*flatent;
    FILE    *finlatent;
    FILE *fscore;
    FILE *fpsipos;
    FILE *fpsineg;

  STRUCTMODEL model;
  STRUCT_LEARN_PARM sparm;

  SAMPLE testsample;
  
  LATENT_VAR h;

  /* read input parameters */
  read_input_parameters(argc,argv,testfile,modelfile,labelfile,latentfile,inlatentfile,scorefile,psiposfile,psinegfile,&log_psis,&sparm);

    printf("%f\n",sparm.C);
	flabel = fopen(labelfile,"w");
	flatent = fopen(latentfile,"w");
        fscore = fopen(scorefile, "w");
        finlatent = NULL;
        fpsipos = NULL;
        fpsineg = NULL;

        if (log_psis) {
          fpsipos = fopen(psiposfile, "w");
          fpsineg = fopen(psinegfile, "w");
        }

  init_struct_model(get_sample_size(testfile), KERNEL_INFO_FILE, &model);

  read_struct_model(modelfile, &model);

  IMAGE_KERNEL_CACHE ** cached_images = init_cached_images(&model);

  /* read test examples */
	printf("Reading test examples..."); fflush(stdout);
	testsample = read_struct_examples(testfile, &model, &sparm);
	printf("done.\n");

  
  avghingeloss = 0.0;
  correct = 0;
  weighted_correct=0.0;
  int impute = (int) (finlatent == NULL);
  int *valid_example_kernel = (int *) malloc(5*sizeof(int));
  for(i = 0; i < model.num_kernels; i++)
    valid_example_kernel[i] = 1;
  printf("IS PLANNING TO IMPUTE %d\n",impute);
  double total_example_weight = 0;
  for (i=0;i<testsample.n;i++) {
    if(finlatent) {
        read_latent_var(&h,finlatent);
        //printf("%d %d\n",h.position_x,h.position_y);
    }
    //printf("%f\n",sparm.C);
    double pos_score = classify_struct_example(testsample.examples[i].x,&y,&h,cached_images,&model,&sparm,impute);


		total_example_weight += testsample.examples[i].x.example_cost;
		double hinge_l = get_hinge_l_from_pos_score(pos_score,testsample.examples[i].y);
		printf("with a pos_score of %f, a label of %d we get a hinge_l of %f\n", pos_score, testsample.examples[i].y, hinge_l);
    double weighted_hinge_l = hinge_l * testsample.examples[i].x.example_cost;
    avghingeloss += weighted_hinge_l;
  	if (hinge_l<1) {
			correct++;
			weighted_correct+=testsample.examples[i].x.example_cost;
		}

		LABEL guesslabel;
		if(pos_score>0)
			guesslabel.label=1;
		else
			guesslabel.label=0;
		print_label(guesslabel,flabel);
		fprintf(flabel,"\n"); fflush(flabel);

		print_latent_var(testsample.examples[i].x, h,flatent);

     char * img_num_str = testsample.examples[i].x.image_path;
     fprintf(fscore, "%s %f\n", img_num_str, pos_score); fflush(fscore);

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
  printf("Objective Value with C=%f is %f\n\n\n", sparm.C, (sparm.C * avghingeloss) + w_cost);
  printf("Average hinge loss on dataset: %.4f\n", avghingeloss);
  printf("Zero/one error on test set: %.4f\n", 1.0 - ((float) correct)/testsample.n);
  printf("Weighted zero/one error on the test set %.4f\n", 	1.0 - (weighted_correct/total_example_weight));

  printf("zeroone %.4f weightedzeroone %.4f\n", 1.0 - ((float) correct)/testsample.n, 1.0 - (weighted_correct/total_example_weight));  

  fclose(fscore);

  if (log_psis) {
    fclose(fpsipos);
    fclose(fpsineg);
  }

  free_cached_images(cached_images, &model);
  free_struct_sample(testsample);
  free_struct_model(model,&sparm);

  return(0);

}


void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, char *labelfile, char *latentfile, char *inlatentfile, char *scorefile, char *psiposfile, char *psinegfile, int *log_psis, STRUCT_LEARN_PARM *sparm) {

  long i;
  
  /* set default */
  strcpy(modelfile, "lssvm_model");
  strcpy(labelfile, "lssvm_label");
  strcpy(latentfile, "lssvm_latent");
  strcpy(scorefile, "lssvm_score");
  strcpy(inlatentfile,"lssvm_inlatent");
  strcpy(psiposfile, "");
  strcpy(psinegfile, "");
  *log_psis = 0;
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
	        strcpy(scorefile,argv[i+4]);
        if(i+5<argc)
          strcpy(psiposfile,argv[i+5]);
        if(i+6<argc)
          strcpy(psinegfile,argv[i+6]);
        //if(i+5<argc)
      //strcpy(inlatentfile,argv[i+5]);
        //else
   inlatentfile[0] = '\0';
   assert((psiposfile[0] != '\0') == (psinegfile[0] != '\0')); //either both or neither of psiposfile and psinegfile are given
   if (psiposfile[0] != '\0') {
     *log_psis = 1;
   }
   parse_struct_parameters(sparm);
}
