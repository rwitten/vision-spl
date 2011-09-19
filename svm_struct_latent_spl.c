/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_spl.c                                            */
/*                                                                      */
/*   Main Optimization Code for Latent SVM^struct using Self-Paced      */
/*   Learning. NOTE: This implementation modifies the CCCP code by      */
/*   Chun-Nam Yu, specifically the file svm_struct_latent_cccp.c,       */
/*   which is a part of the Latent SVM^struct package available on      */
/*   Chun-Nam Yu's webpage.                                             */
/*                                                                      */
/*   Authors: M. Pawan Kumar and Ben Packer                             */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <unistd.h>
#include <pthread.h>

#include "./SFMT-src-1.3.3/SFMT.h"
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "svm_struct_latent_api.h"
#include "./svm_light/svm_learn.h"


#define ITERS_TO_UPDATE_LOWER_BOUND 10
#define ALPHA_THRESHOLD 1E-8
#define IDLE_ITER 25
#define CLEANUP_CHECK 50
#define STOP_PREC 1E-2
#define UPDATE_BOUND 3
#define MAX_CURRICULUM_ITER 10
#define NUM_THREADS 0
#define MAX_OUTER_ITER 20
#define SSG_PRINT_ITERS 100

#define MAX(x,y) ((x) < (y) ? (y) : (x))
#define MIN(x,y) ((x) > (y) ? (y) : (x))

#define DEBUG_LEVEL 0

#define KERNEL_INFO_FILE "data/kernel_info.txt"

int mosek_qp_optimize(double**, double*, double*, long, double, double*);

void my_read_input_parameters(int argc, char* argv[], char *trainfile, char *modelfile, char *examplesfile, char *timefile, char *latentfile,LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *struct_parm, double *init_spl_weight, double *spl_factor);

void my_wait_any_key();

int resize_cleanup(int size_active, int **ptr_idle, double **ptr_alpha, double** ptr_alpha_lb, double **ptr_delta, double **ptr_delta_plus_A_wlast, DOC ***ptr_dXc,
		double ***ptr_G, int *mv_iter);

void approximate_to_psd(double **G, int size_active, double eps);

void Jacobi_Cyclic_Method(double eigenvalues[], double *eigenvectors, double *A, int n);

double sprod_nn(double *a, double *b, long n) {
  double ans=0.0;
  long i;
  for (i=1;i<n+1;i++) {
    ans+=a[i]*b[i];
  }
  return(ans);
}

void add_vector_nn(double *w, double *dense_x, long n, double factor) {
  long i;
  for (i=1;i<n+1;i++) {
    w[i]+=factor*dense_x[i];
  }
}

double* add_list_nn(SVECTOR *a, long totwords) 
     /* computes the linear combination of the SVECTOR list weighted
	by the factor of each SVECTOR. assumes that the number of
	features is small compared to the number of elements in the
	list */
{
    SVECTOR *f;
    long i;
    double *sum;

    sum=create_nvector(totwords);

    for(i=0;i<=totwords;i++) 
      sum[i]=0;

    for(f=a;f;f=f->next)  
      add_vector_ns(sum,f,f->factor);

    return(sum);
}

void find_most_violated_constraint(EXAMPLE *ex, LABEL *ybar, LATENT_VAR *hbar, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  switch (sparm->margin_type) {
  case 0: find_most_violated_constraint_marginrescaling (ex->x, ex->h, ex->y, ybar, hbar, cached_images, valid_kernels, sm, sparm); break;
  case 1: find_most_violated_constraint_differenty (ex->x, ex->h, ex->y, ybar, hbar, cached_images, valid_kernels, sm, sparm); break;
  default: printf ("Unrecognized margin_type '%d'\n", sparm->margin_type);
    exit(1);
  }
}

double print_all_scores(EXAMPLE *ex, SVECTOR **fycache, long m, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, double C, int *valid_examples, int ** valid_example_kernels) {

  long i;
  SVECTOR *f, *fy, *fybar, *lhs;
  LABEL       ybar;
  LATENT_VAR hbar = make_latent_var(sm);
  double lossval, margin;
  double *new_constraint;
	double obj = 0.0;

  /* find cutting plane */
  lhs = NULL;
  margin = 0;
  for (i=0;i<m;i++) {
	if(!valid_examples[i])
		continue;
    find_most_violated_constraint(&(ex[i]), &ybar, &hbar, cached_images, valid_example_kernels[i], sm, sparm);
    /* get difference vector */
    fy = psi(ex[i].x,ex[i].y,ex[i].h,cached_images, valid_example_kernels[i],sm,sparm);
    fybar = psi(ex[i].x,ybar,hbar,cached_images,valid_example_kernels[i],sm,sparm);
    lossval = loss(ex[i].y,ybar,hbar,sparm);
    printf("%f\n",lossval+sprod_ns(sm->w,fybar)-sprod_ns(sm->w,fy));

    /* scale difference vector */
    for (f=fy;f;f=f->next) {
      //f->factor*=1.0/m;
      f->factor*=ex[i].x.example_cost/m;
    }
    for (f=fybar;f;f=f->next) {
      //f->factor*=-1.0/m;
      f->factor*=-ex[i].x.example_cost/m;
    }
    /* add ybar to constraint */
    append_svector_list(fy,lhs);
    append_svector_list(fybar,fy);
    lhs = fybar;
    //margin+=lossval/m;
		margin += lossval*ex[i].x.example_cost/m;
  }

  /* compact the linear representation */
  new_constraint = add_list_nn(lhs, sm->sizePsi+1);
  free_svector(lhs);

	obj = margin;
	for(i = 1; i < sm->sizePsi+2; i++)
		obj -= new_constraint[i]*sm->w[i];
	if(obj < 0.0)
		obj = 0.0;
	obj *= C;
	for(i = 1; i < sm->sizePsi+2; i++)
		obj += 0.5*sm->w[i]*sm->w[i];
  free(new_constraint);

	return obj;
}


double current_obj_val(EXAMPLE *ex, SVECTOR **fycache, long m, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, double C, int *valid_examples, int ** valid_example_kernels) {

  long i;
  SVECTOR *f, *fy, *fybar, *lhs;
  LABEL       ybar;
  LATENT_VAR hbar = make_latent_var(sm);
  double lossval, margin;
  double *new_constraint;
	double obj = 0.0;

  /* find cutting plane */
  lhs = NULL;
  margin = 0;
  for (i=0;i<m;i++) {
    if(!valid_examples[i])
        continue;
    find_most_violated_constraint(&(ex[i]), &ybar, &hbar, cached_images, valid_example_kernels[i], sm, sparm);
    /* get difference vector */
    fy = psi(ex[i].x,ex[i].y,ex[i].h,cached_images, valid_example_kernels[i],sm,sparm);
    fybar = psi(ex[i].x,ybar,hbar,cached_images,valid_example_kernels[i],sm,sparm);
    lossval = loss(ex[i].y,ybar,hbar,sparm);

    /* scale difference vector */
    for (f=fy;f;f=f->next) {
      //f->factor*=1.0/m;
      f->factor*=ex[i].x.example_cost/m;
    }
    for (f=fybar;f;f=f->next) {
      //f->factor*=-1.0/m;
      f->factor*=-ex[i].x.example_cost/m;
    }
    /* add ybar to constraint */
    append_svector_list(fy,lhs);
    append_svector_list(fybar,fy);
    lhs = fybar;
    //margin+=lossval/m;
		margin += lossval*ex[i].x.example_cost/m;
  }

  /* compact the linear representation */
  new_constraint = add_list_nn(lhs, sm->sizePsi+1);
  free_svector(lhs);
	free_latent_var(hbar);
	obj = margin;
	for(i = 1; i < sm->sizePsi+2; i++)
		obj -= new_constraint[i]*sm->w[i];
	if(obj < 0.0)
		obj = 0.0;
	obj *= C;
	for(i = 1; i < sm->sizePsi+2; i++)
		obj += 0.5*sm->w[i]*sm->w[i];
  free(new_constraint);

	return obj;
}

int compar(const void *a, const void *b)
{
  sortStruct *c = (sortStruct *) a;
  sortStruct *d = (sortStruct *) b;
  if(c->val < d->val)
    return -1;
  if(c->val > d->val)
    return 1;
  return 0;
}

void* handle_psi(void* inputa)
{
    psi_job* background  = (psi_job*) inputa;

    int more_work_to_do = 1;
    while(more_work_to_do)
    {
       pthread_mutex_lock(background->curr_lock);
       int curr_task = *(background->curr_task);
        *(background->curr_task)= curr_task + 1;
        pthread_mutex_unlock(background->curr_lock);

        if(curr_task>=background->m) //we're done
        {
            more_work_to_do = 0;
        }
        else  //have to do the job
        {
            background->output_vectors[curr_task] = psi((background->ex_list[curr_task].x), (background->ybar_list[curr_task]), (background->hbar_list[curr_task]), background->cached_images, background->valid_example_kernels[curr_task], background->sm, background->sparm); 

             pthread_mutex_lock(background->completed_lock);
             int completed_tasks = *(background->completed_tasks);
             *(background->completed_tasks)= completed_tasks + 1;
             pthread_mutex_unlock(background->completed_lock);
	    }
    }
   if(NUM_THREADS)
	   pthread_exit(0);
	 else
	 	 return NULL;
}

void* handle_fmvc(void* inputa)
{
    fmvc_job* background  = (fmvc_job*) inputa;

    int more_work_to_do = 1;
    while(more_work_to_do)
    {
       pthread_mutex_lock(background->curr_lock);
       int curr_task = *(background->curr_task);
        *(background->curr_task)= curr_task + 1;
        pthread_mutex_unlock(background->curr_lock);

        if(curr_task>=background->m) //we're done
        {
            more_work_to_do = 0;
        }
        else  //have to do the job
        {
          if (background->valid_examples[curr_task]) {
            find_most_violated_constraint(&(background->ex_list[curr_task]), &(background->ybar_list[curr_task]), &(background->hbar_list[curr_task]), background->cached_images, background->valid_example_kernels[curr_task], background->sm, background->sparm);
          }
             pthread_mutex_lock(background->completed_lock);
             int completed_tasks = *(background->completed_tasks);
             *(background->completed_tasks)= completed_tasks + 1;
             pthread_mutex_unlock(background->completed_lock);
	    }
    }
   if(NUM_THREADS)
	   pthread_exit(0);
	 else
	 	 return NULL;
}

void find_psi_parallel(int m,EXAMPLE* ex_list, LABEL* ybar_list, LATENT_VAR* hbar_list, IMAGE_KERNEL_CACHE ** cached_images, int ** valid_example_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, SVECTOR** output_vectors)
{
    pthread_t mythreads[NUM_THREADS];
    int curr_task = 0;
    int completed_tasks = 0;

    pthread_mutex_t completed_lock;
    pthread_mutex_init(&completed_lock, NULL);
    pthread_mutex_t curr_lock;
    pthread_mutex_init(&curr_lock, NULL);

    psi_job background;
    background.valid_example_kernels = valid_example_kernels;
    background.m = m;
    background.curr_task = &curr_task;
    background.completed_tasks = &completed_tasks;
    background.curr_lock = &curr_lock;
    background.completed_lock = &completed_lock;
    background.ex_list = ex_list;
    background.ybar_list = ybar_list;
    background.hbar_list = hbar_list;
    background.output_vectors = output_vectors;
    background.cached_images = cached_images;
    background.sm = sm;
    background.sparm = sparm;
    int i;
    for(i=0; i < NUM_THREADS; i++)
    {
        pthread_create(&mythreads[i], NULL, handle_psi, &background);
    }

    if(NUM_THREADS)
    {
        for(i=0; i<NUM_THREADS;i++)
        {
             pthread_join(mythreads[i],NULL);
        }
    }
    else
    {
        handle_psi(&background);
    }
}

void find_most_violated_constraint_parallel(int m,EXAMPLE* ex_list, LABEL* ybar_list, LATENT_VAR* hbar_list, IMAGE_KERNEL_CACHE ** cached_images, int * valid_examples, int ** valid_example_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
    pthread_t mythreads[NUM_THREADS];
    int curr_task = 0;
    int completed_tasks = 0;

    pthread_mutex_t completed_lock;
    pthread_mutex_init(&completed_lock, NULL);
    pthread_mutex_t curr_lock;
    pthread_mutex_init(&curr_lock, NULL);

    fmvc_job background;
    background.valid_example_kernels = valid_example_kernels;
    background.valid_examples = valid_examples;
    background.m = m;
    background.curr_task = &curr_task;
    background.completed_tasks = &completed_tasks;
    background.curr_lock = &curr_lock;
    background.completed_lock = &completed_lock;
    background.ex_list = ex_list;
    background.ybar_list = ybar_list;
    background.hbar_list = hbar_list;
    background.cached_images = cached_images;
    background.sm = sm;
    background.sparm = sparm;
    int i;
    for(i=0; i < NUM_THREADS; i++)
    {
        pthread_create(&mythreads[i], NULL, handle_fmvc, &background);
    }

    if(NUM_THREADS)
    {
        for(i=0; i<NUM_THREADS;i++)
        {
             pthread_join(mythreads[i],NULL);
        }
    }
    else
    {
        handle_fmvc(&background);
    }
}

SVECTOR* find_cutting_plane(EXAMPLE *ex, SVECTOR **fycache, double *margin, long m, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples, int** valid_example_kernels) {

  long i;
  SVECTOR *f, *lhs;
  SVECTOR** fys = (SVECTOR**) malloc(m*sizeof(SVECTOR*));
  SVECTOR** fybars = (SVECTOR**) malloc(m*sizeof(SVECTOR*));
  LABEL       ybar;
  double lossval;
  double *new_constraint;
	long valid_count = 0;

  long l,k;
  SVECTOR *fvec;
  WORD *words;  

  /* find cutting plane */
  lhs = NULL;
  *margin = 0;

   for (i=0;i<m;i++) {
        if (valid_examples[i]) {
             valid_count++;
        }
   }

  LABEL*       ybar_list =(LABEL*) ( malloc(m*sizeof(LABEL)));
  LATENT_VAR* hbar_list = (LATENT_VAR*)(malloc(m*sizeof(LATENT_VAR)));
  for(int i = 0; i < m ; i++)
    hbar_list[i] = make_latent_var(sm);


  find_most_violated_constraint_parallel(m,ex, ybar_list, hbar_list, cached_images,valid_examples, valid_example_kernels,  sm, sparm);

  LABEL*       y_list =(LABEL*) ( malloc(m*sizeof(LABEL)));
  LATENT_VAR* h_list = (LATENT_VAR*)(malloc(m*sizeof(LATENT_VAR)));
  for(int i = 0 ; i < m ; i++)
  {
    h_list[i] = ex[i].h;//shallow copy of latent vars so don't screw with it.
    y_list[i] = ex[i].y;
  }
  find_psi_parallel(m, ex, y_list, h_list, cached_images, valid_example_kernels, sm, sparm,fys);
  find_psi_parallel(m, ex, ybar_list, hbar_list, cached_images, valid_example_kernels, sm, sparm,fybars);

  for (i=0;i<m;i++) {

        if (!valid_examples[i]) {
            continue;
        }


    lossval = loss(ex[i].y,ybar_list[i],hbar_list[i],sparm);
    //printf("for image %d loss is %f\n", i, lossval);
    free_label(ybar);
		
    /* scale difference vector */
    for (f=fys[i];f;f=f->next) {
      //f->factor*=1.0/m;
      //f->factor*=ex[i].x.example_cost/m;
      f->factor*=ex[i].x.example_cost/valid_count;
    }
    for (f=fybars[i];f;f=f->next) {
      //f->factor*=-1.0/m;
      //f->factor*=-ex[i].x.example_cost/m;
      f->factor*=-ex[i].x.example_cost/valid_count;
    }
    /* add ybar to constraint */
    append_svector_list(fys[i],lhs);
    append_svector_list(fybars[i],fys[i]);
    lhs = fybars[i];
    //*margin+=lossval/m;
    //*margin+=lossval*ex[i].x.example_cost/m;
    *margin+=lossval*ex[i].x.example_cost/valid_count;
  }
  free(ybar_list);
  free(y_list);
  free(h_list);
  
  /*for(int i = 0 ; i < m ; i++)
  {
    free(fys[i]);
    free(fybars[i]);
  }*/
  free(fys);
  free(fybars);
  for(int i = 0 ; i < m ; i++)
	free_latent_var(hbar_list[i]);
  free(hbar_list);
  /* compact the linear representation */
  new_constraint = add_list_nn(lhs, sm->sizePsi+1);
  free_svector(lhs);

  l=0;
  for (i=1;i<sm->sizePsi+2;i++) {
    if (fabs(new_constraint[i])>1E-10) l++; // non-zero
  }
  words = (WORD*)my_malloc(sizeof(WORD)*(l+1)); 
  assert(words!=NULL);
  k=0;
  for (i=1;i<sm->sizePsi+2;i++) {
    if (fabs(new_constraint[i])>1E-10) {
      words[k].wnum = i;
      words[k].weight = new_constraint[i]; 
      k++;
    }
  }
  words[k].wnum = 0;
  words[k].weight = 0.0;
  fvec = create_svector(words,(char*)"",1);

  free(words);
  free(new_constraint);

  return(fvec); 
}

/* project weights to ball of radius 1/sqrt{lambda} */
void project_weights(double *w, int sizePsi, double C)
{
	double norm_squared = 0.0;
	double projection_factor = 1.0;
	int i;
	for(i=0;i<sizePsi+2;i++)
		norm_squared += w[i]*w[i];
	if(.5 * norm_squared > C)
	{
		projection_factor = sqrt((2*C)/(norm_squared));
		for(i=0;i<=sizePsi+2;i++)
			w[i] = w[i]*projection_factor;
	}
//    norm_squared = 0.0;
//    for(i = 0 ; i  < sizePsi +2 ; i++)
//       norm_squared += w[i]*w[i];

//    printf("New obj is %f\n", .5*norm_squared);
}

long *randperm(long m, long n)
{
	long *perm, *map;
	long i,j;

	if(m < n)
		n = m;
  perm = (long *) malloc(sizeof(long)*n);
	if(m == n) {
		for(i = 0; i < m; i++)
			perm[i] = i;
		return perm;
	}
  map = (long *) malloc(sizeof(long)*m);
  for(i = 0; i < m; i++)
    map[i] = i;
  for(i = 0; i < n; i++)
  {
    int r = (int) (((double) m-i)*((double) rand())/(RAND_MAX+1.0));
    perm[i] = map[r];
    for(j = r; j < m-1; j++)
      map[j] = map[j+1];
  }
  free(map);
  return perm;
}

void pick_ssg_set(int* valid_examples, int* temp_valid_examples, int m, double frac)
{
    for(int i = 0; i < m ; i++)
    {
        temp_valid_examples[i] = valid_examples[i] * (genrand_res53()<frac ? 1 : 0);
    }
}

/* stochastic subgradient descent for solving the convex structural SVM problem */
double stochastic_subgradient_descent(double *w, long m, int MAX_ITER, double C, double epsilon, SVECTOR **fycache, EXAMPLE *ex, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples, int** valid_example_kernels) {
	/* constants */
  printf("Running stochastic structural SVM solver: "); fflush(stdout); 


  /* initializations */
  int iter = 0;
  srand(time(NULL));
  double best_obj=C; //since zero gets C
  double lambda = 1/(m*C);
  double* grad = (double*) malloc(sizeof(double)*(2+sm->sizePsi));
  printf("the relation is lambda %f m %ld and C %f\n", lambda, m, C);
  int* temp_valid_examples = (int*) malloc(sizeof(int)*m);
//	clear_nvector(w,sm->sizePsi);
	while(iter<MAX_ITER) {
        double prox_reg = 1e2 * ( 1e4 + iter);
        struct timeval start_time;
        struct timeval finish_time;
        gettimeofday(&start_time, NULL);
        printf("."); fflush(stdout);
        double margin;
        pick_ssg_set(valid_examples,temp_valid_examples,m, 1);
        SVECTOR* new_constraint;
        if(iter % SSG_PRINT_ITERS != 0)
            new_constraint = find_cutting_plane(ex, fycache, &margin, m, cached_images, sm, sparm, temp_valid_examples, valid_example_kernels);
        else
            new_constraint = find_cutting_plane(ex, fycache, &margin, m, cached_images, sm, sparm, valid_examples,valid_example_kernels);

        for(int i = 0 ;  i < sm->sizePsi +2; i++)
            grad[i] = prox_reg * w[i];

        if(margin>0) 
            add_vector_ns(grad, new_constraint, C);


        if(iter % SSG_PRINT_ITERS == 0)
        {
            double objval = C*(margin - sprod_ns(w, new_constraint));
            if(objval<0)
                objval=0;
            for( int i = 0 ; i < sm->sizePsi+2 ; i ++)
                objval += .5* w[i]*w[i];

            if(best_obj > objval)
            {
                best_obj = objval;
            }
            printf("new objective is %f with best %f on iter %d\n", objval, best_obj,iter );
        }

        for (int i = 0 ; i < sm->sizePsi + 2; i++)
            w[i] = grad[i]/(1+prox_reg);

        project_weights(w, sm->sizePsi, best_obj);
        iter+=1;
        free_svector(new_constraint);
        gettimeofday(&finish_time, NULL);
//        double microseconds = 1e6 * (int)(finish_time.tv_sec - start_time.tv_sec) + (int)(finish_time.tv_usec - start_time.tv_usec);
//        printf("SSG step took %f milliseconds\n", microseconds/1000);
 	}
    free(temp_valid_examples);

  printf(" Inner loop optimization finished.\n"); fflush(stdout); 

	/* return primal objective value */
	return(best_obj);
}

double cutting_plane_algorithm(double *w, long m, int MAX_ITER, double C, double epsilon, SVECTOR **fycache, EXAMPLE *ex, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples, int** valid_example_kernels) {
  long i,j;
  double *alpha;
  double *alpha_lb;
  DOC **dXc; /* constraint matrix */
  double *delta; /* rhs of constraints */
  double * delta_plus_A_wlast;
	SVECTOR *new_constraint;
  int iter, size_active; 
  double value;
  double margin;
  double primal_obj, cur_obj;
	double *cur_slack = NULL;
	int mv_iter;
	int *idle = NULL;
	double **G = NULL;
	SVECTOR *f;
	int r;

  double* w_last = (double*)calloc(sm->sizePsi + 2, sizeof(double));
  double* w_best = (double*)calloc(sm->sizePsi + 2, sizeof(double));
  memcpy(w_last, w, (sm->sizePsi + 2) * sizeof(double));

  /* set parameters for hideo solver */
  LEARN_PARM lparm;
  KERNEL_PARM kparm;
  MODEL *svm_model=NULL;
  lparm.biased_hyperplane = 0;
  lparm.epsilon_crit = MIN(epsilon,0.001);
  lparm.svm_c = C;
  lparm.sharedslack = 1;
  kparm.kernel_type = LINEAR;

  lparm.remove_inconsistent=0;
  lparm.skip_final_opt_check=0;
  lparm.svm_maxqpsize=10;
  lparm.svm_newvarsinqp=0;
  lparm.svm_iter_to_shrink=-9999;
  lparm.maxiter=100000;
  lparm.kernel_cache_size=40;
  lparm.eps = MIN(.00001,epsilon); 
  lparm.transduction_posratio=-1.0;
  lparm.svm_costratio=1.0;
  lparm.svm_costratio_unlab=1.0;
  lparm.svm_unlabbound=1E-5;
  lparm.epsilon_a=1E-10;  /* changed from 1e-15 */
  lparm.compute_loo=0;
  lparm.rho=1.0;
  lparm.xa_depth=0;
  strcpy(lparm.alphafile,"");
  kparm.poly_degree=3;
  kparm.rbf_gamma=1.0;
  kparm.coef_lin=1;
  kparm.coef_const=1;
  strcpy(kparm.custom,"empty");
 
  iter = 0;
  size_active = 0;
  alpha = NULL;
	alpha_lb = NULL;
  dXc = NULL;
  delta = NULL;
	delta_plus_A_wlast = NULL;

  printf("Running structural SVM solver: "); fflush(stdout); 
  
  struct timeval start_time;
  struct timeval finish_time;
  gettimeofday(&start_time, NULL);

  new_constraint = find_cutting_plane(ex, fycache, &margin, m, cached_images, sm, sparm, valid_examples, valid_example_kernels);
 	value = margin - sprod_ns(w, new_constraint);
	printf("First violation is %f\n", value);

	double UPPER_BOUND = DBL_MAX;
	double LOWER_BOUND = 0;
	double* w_temp = (double*) malloc(sizeof(double) *(sm->sizePsi+2));
	while(((UPPER_BOUND-LOWER_BOUND>C*epsilon)&&(iter<MAX_ITER))) {
		iter+=1;
		size_active+=1;

		printf("."); fflush(stdout); 
		gettimeofday(&finish_time, NULL);

		int million = 1000000;
		double microseconds = million * (finish_time.tv_sec - start_time.tv_sec) + (finish_time.tv_usec - start_time.tv_usec);
		printf("Cutting plane took %f milliseconds.\n", microseconds / 1000.0);
		start_time.tv_sec = finish_time.tv_sec;
		start_time.tv_usec = finish_time.tv_usec;

    /* add  constraint */
        dXc = (DOC**)realloc(dXc, sizeof(DOC*)*size_active);
        assert(dXc!=NULL);
        dXc[size_active-1] = (DOC*)malloc(sizeof(DOC));
        dXc[size_active-1]->fvec = new_constraint; 
        dXc[size_active-1]->slackid = 1; // only one common slackid (one-slack)
        dXc[size_active-1]->costfactor = 1.0;

        delta = (double*)realloc(delta, sizeof(double)*size_active);
        assert(delta!=NULL);
        delta[size_active-1] = margin;
        delta_plus_A_wlast = (double*)realloc(delta_plus_A_wlast, sizeof(double)*size_active);
        assert(delta_plus_A_wlast != NULL);

        for(i = 0 ; i < size_active ; i ++)
            delta_plus_A_wlast[i] = delta[i] - ((2.0 * sparm->prox_weight) / (1.0 + 2.0 * sparm->prox_weight)) * sprod_ns(w_last, dXc[i]->fvec);
            
        alpha = (double*)realloc(alpha, sizeof(double)*size_active);
        assert(alpha!=NULL);
        alpha[size_active-1] = 0.0;
        alpha_lb = (double*)realloc(alpha_lb, sizeof(double)*size_active);
        assert(alpha_lb!=NULL);
        alpha_lb[size_active-1] = 0.0;

		idle = (int *) realloc(idle, sizeof(int)*size_active);
		assert(idle!=NULL);
		idle[size_active-1] = 0;

		/* update Gram matrix */
		G = (double **) realloc(G, sizeof(double *)*size_active);
		assert(G!=NULL);
		G[size_active-1] = NULL;
		for(j = 0; j < size_active; j++) {
			G[j] = (double *) realloc(G[j], sizeof(double)*size_active);
			assert(G[j]!=NULL);
		}
		for(j = 0; j < size_active-1; j++) {
			G[size_active-1][j] = sprod_ss(dXc[size_active-1]->fvec, dXc[j]->fvec);
			G[j][size_active-1]  = G[size_active-1][j];
		}
		G[size_active-1][size_active-1] = sprod_ss(dXc[size_active-1]->fvec,dXc[size_active-1]->fvec);
		/* hack: add a constant to the diagonal to make sure G is PSD */
		G[size_active-1][size_active-1] += 1e-6;

   	/* solve QP to update alpha */
		struct timeval qp_start_time;
		struct timeval qp_finish_time;
		gettimeofday(&qp_start_time, NULL);
        r = mosek_qp_optimize(G, delta_plus_A_wlast, alpha, (long) size_active, C / (1.0 + 2.0 * sparm->prox_weight), &cur_obj); // regularized solve
		gettimeofday(&qp_finish_time, NULL);
		microseconds = 1e6 * (qp_finish_time.tv_sec - qp_start_time.tv_sec) + (qp_finish_time.tv_usec - qp_start_time.tv_usec);
		printf("Solving QP took %f\n", (double)microseconds/1000.0);
    
        if(r >= 1293 && r <= 1296)
        {
            printf("r:%d. G might not be psd due to numerical errors.\n",r);
            exit(1);
        }
        else if(r)
        {
            printf("Error %d in mosek_qp_optimize: Check ${MOSEKHOME}/${VERSION}/tools/platform/${PLATFORM}/h/mosek.h\n",r);
            exit(1);
        }

        clear_nvector(w,sm->sizePsi+1);
        for (j=0;j<size_active;j++) {
            if (alpha[j] > C * ALPHA_THRESHOLD / (1.0 + 2.0 * sparm->prox_weight)) {
                    add_vector_ns(w,dXc[j]->fvec,alpha[j]);
                    idle[j] = 0;
            }
                else
                    idle[j]++;
        }

		for (j = 0; j < sm->sizePsi +2; ++j) {
			w[j] += ((2.0 * sparm->prox_weight) / (1.0 + 2.0 * sparm->prox_weight)) * w_last[j];
		}
							

		cur_slack = (double *) realloc(cur_slack,sizeof(double)*size_active);

		for(i = 0; i < size_active; i++) {
			cur_slack[i] = 0.0;
			for(f = dXc[i]->fvec; f; f = f->next) {
				j = 0;
				while(f->words[j].wnum) {
					cur_slack[i] += w[f->words[j].wnum]*f->words[j].weight;
					j++;
				}
			}
			if(cur_slack[i] >= delta[i])
				cur_slack[i] = 0.0;
			else
				cur_slack[i] = delta[i]-cur_slack[i];
		}

		mv_iter = 0;
		if(size_active > 1) {
			for(j = 0; j < size_active; j++) {
				if(cur_slack[j] >= cur_slack[mv_iter])
					mv_iter = j;
			}
		}


 		new_constraint = find_cutting_plane(ex, fycache, &margin, m, cached_images, sm, sparm, valid_examples, valid_example_kernels);

		double threshold = C*(margin - sprod_ns(w, new_constraint));
        if(threshold<0)
            threshold=0;

		for( i = 0 ; i < sm->sizePsi+2 ; i ++)
			threshold += .5* w[i]*w[i];

		if(threshold<UPPER_BOUND) //THIS IS THE BOUND WE GET BY EVALUATING CURRENT SOLUTION ON THE FULL QP
        {
			UPPER_BOUND=threshold;
            memcpy(w_best, w, sizeof(double)*(sm->sizePsi+2));
        }

        if( (iter % ITERS_TO_UPDATE_LOWER_BOUND) == 0)
        {
            double lb1;
            r = mosek_qp_optimize(G, delta, alpha_lb, (long) size_active, C, &lb1); // unregularized solve
            if(r >= 1293 && r <= 1296)
            {
                printf("r:%d. G might not be psd due to numerical errors.\n",r);
                exit(1);
            }
            else if(r)
            {
                printf("Error %d in mosek_qp_optimize: Check ${MOSEKHOME}/${VERSION}/tools/platform/${PLATFORM}/h/mosek.h\n",r);
                exit(1);
            }
            clear_nvector(w_temp,sm->sizePsi+1);
            for (j=0;j<size_active;j++) {
                if (alpha_lb[j] > C * ALPHA_THRESHOLD) {
                        add_vector_ns(w_temp,dXc[j]->fvec,alpha_lb[j]);
                }
            }
            double max_slack = -DBL_MAX;
            assert(size_active>1);
            for(i = 0; i < size_active; i++) {
                double slack = 0;
                for(f = dXc[i]->fvec; f; f = f->next) {
                    j = 0;
                    while(f->words[j].wnum) {
                        slack += w_temp[f->words[j].wnum]*f->words[j].weight;
                        j++;
                    }
                }
                if(slack >= delta[i])
                    slack = 0;
                else
                    slack = delta[i]-slack;
                if(slack>max_slack)
                    max_slack = slack;
            }

            double lb = C*max_slack;
            for(i = 0 ; i < sm->sizePsi+2 ; i ++)
                lb += .5* w_temp[i]*w_temp[i];
            //printf("New lower bound is %f\n", lb);
            if(lb > LOWER_BOUND) // THIS EXACT SOLUTION OF PARTIAL QP PROVIDES A LOWER BOUND 
                                //(SINCE IT HAS FEWER CONSTRAINTS THAN THE TRUE PROBLEM
            {
                LOWER_BOUND=lb;
            }
        }

		if((iter % CLEANUP_CHECK) == 0)
		{
			printf("+"); fflush(stdout);
			size_active = resize_cleanup(size_active, &idle, &alpha, &alpha_lb,&delta, &delta_plus_A_wlast, &dXc, &G, &mv_iter);
		}
		printf("We need to get %f less than %f on iter %d\n", UPPER_BOUND, LOWER_BOUND+C*epsilon, iter);
	    memcpy(w_last, w, (sm->sizePsi + 2) * sizeof(double));
        assert(UPPER_BOUND + epsilon>LOWER_BOUND);
 	} // end cutting plane while loop 

	primal_obj = current_obj_val(ex, fycache, m, cached_images, sm, sparm, C, valid_examples, valid_example_kernels);

  printf(" Inner loop optimization finished.\n"); fflush(stdout); 
      
  /* free memory */
  for (j=0;j<size_active;j++) {
	free(G[j]);
    free_example(dXc[j],1);	
  }
  memcpy(w,w_best, sizeof(double)*(sm->sizePsi+2));

  free(w_best);
  free(w_temp);
  free(G);
  free(dXc);
  free(alpha);
  free(w_last);
  free(delta_plus_A_wlast);
  free(delta);
  free_svector(new_constraint);
  free(cur_slack);
  free(idle);
  if (svm_model!=NULL) free_model(svm_model,0);

  return(primal_obj);
}

int check_acs_convergence(int *prev_valid_examples, int *valid_examples, int** prev_valid_example_kernels, int** valid_example_kernels, long m, int kernel_size)
{
	long i,k;
	int converged = 1;

	for (i=0;i<m;i++) {
		if (prev_valid_examples[i] != valid_examples[i]) {
			converged = 0;
			break;
		}
        for ( k=0; k<kernel_size; k++) {
           if(prev_valid_example_kernels[i][k] == valid_example_kernels[i][k]) {
                converged=0;
                break;
            }
        }
	}

	return converged;
}

int update_valid_examples(double *w, long m, double C, SVECTOR **fycache, EXAMPLE *ex, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples, int* kernel_choice, double spl_weight_pos, double spl_weight_neg,int* invalidPositives, int* validPositives) {
	long i;

	for(i=0; i<m;i++)
		valid_examples[i]=0;
    if(!sparm->multi_kernel_spl){
        for(i=0; i<sm->num_kernels;i++) {
            kernel_choice[i]=1;   //since we aren't doing multi_kernel learning, all are on no matter what they said. 
        }
    }

	/* if self-paced learning weight is non-positive, all examples are valid */
	if(spl_weight_neg <= 0.0) {
        assert(spl_weight_pos<=0.0);
		for (i=0;i<m;i++)
			valid_examples[i] = 1;
		printf("DOING CCCP\n");fflush(stdout);
		return (m);
	}
	
	printf("NOT DOING CCCP\n");fflush(stdout);
  int pos_count=0; long j;

	sortStruct *slack = (sortStruct *) malloc(m*sizeof(sortStruct));
	LABEL ybar;
	LATENT_VAR hbar = make_latent_var(sm);
	SVECTOR *f, *fy, *fybar;

	for (i=0;i<m;i++) {
	  find_most_violated_constraint(&(ex[i]), &ybar, &hbar, cached_images, kernel_choice, sm, sparm);
		fy = psi(ex[i].x,ex[i].y,ex[i].h,cached_images,kernel_choice,sm,sparm);
		fybar = psi(ex[i].x,ybar,hbar,cached_images,kernel_choice,sm,sparm);
		slack[i].index = i;
		slack[i].val = loss(ex[i].y,ybar,hbar,sparm);
		for (f=fy;f;f=f->next) {
			j = 0;
			while (1) {
				if(!f->words[j].wnum)
					break;
				slack[i].val -= sm->w[f->words[j].wnum]*f->words[j].weight;
				j++;
			}
		}
		for (f=fybar;f;f=f->next) {
			j = 0;
			while (1) {
				if(!f->words[j].wnum)
					break;
				slack[i].val += sm->w[f->words[j].wnum]*f->words[j].weight;
				j++;
			}
		}
        if(ex[i].y.label)
            pos_count++;
		free_svector(fy);
		free_svector(fybar);
	}

    qsort(slack,pos_count,sizeof(sortStruct),&compar);
    qsort(&slack[pos_count],m-pos_count,sizeof(sortStruct),&compar);
	int nValid = 0;
    
    int pos_cutoff = (int)floor(pos_count*spl_weight_pos + 0.5);
    int neg_cutoff = pos_count+(int)floor((m-pos_count)*spl_weight_neg + 0.5);

    *validPositives = pos_cutoff;
    *invalidPositives = pos_count - pos_cutoff;

    for (i=0; i< pos_cutoff;i++)
    {
        valid_examples[slack[i].index]=1;
    }
    for (i=pos_count; i<neg_cutoff;i++)
    {
        assert(i<m);
        valid_examples[slack[i].index]=1;
    }

	//for (i=0; i<pos_count;i++)
//		printf("%d ", valid_examples[i]);
//	printf("\n");
  nValid = pos_cutoff + (neg_cutoff-pos_count);

	free(slack);
	return nValid;
}

/*can be used to get the single weight for original SPL or to get one of the kernel weights for multi-kernel SPL*/
/*for original SPL, just set all entries in valid_kernels to 1*/
/*for multi-kernel SPL, set only the entry of the kernel you're interested in to 1*/
void get_init_spl_weight(long m, double C, SVECTOR **fycache, EXAMPLE *ex, IMAGE_KERNEL_CACHE ** cached_images, int * valid_kernels, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, double* spl_weight_pos, double* spl_weight_neg) {
	spl_weight_pos[0]= sparm->init_valid_fraction;
    spl_weight_neg[0] = sparm->init_valid_fraction;
    return;
}

double alternate_convex_search(double *w, long m, int MAX_ITER, double C, double epsilon, SVECTOR **fycache, EXAMPLE *ex, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples, int ** valid_example_kernels, double * spl_weight_pos, double* spl_weight_neg) {

    long i, k, j;

    int* nValids = (int*)(calloc(sm->num_kernels,sizeof(int)));
    int* posInvalids = (int*)(calloc(sm->num_kernels,sizeof(int)));
    int* posValids = (int*)(calloc(sm->num_kernels, sizeof(int)));

		int *prev_valid_examples = (int *) malloc(m*sizeof(int));
    int** prev_valid_example_kernels=(int**) malloc(m*sizeof(int*));

    for(i = 0; i<m;i++)
    {
			prev_valid_examples[i]=1;
			prev_valid_example_kernels[i] = (int*)malloc(sm->num_kernels*sizeof(int));
			for(k=0; k<sm->num_kernels;k++)
					prev_valid_example_kernels[i][k] = 1;
		valid_examples[i]=0;
    }

  
    int* this_kernels_examples = (int*) calloc( m, sizeof(int));
    int* kernel_info = (int*) calloc(sm->num_kernels, sizeof(int)); 
    for (i=0;i<sm->num_kernels;i++)
    {
      kernel_info[i] = 1;
			for(j =0; j<m; j++)
				this_kernels_examples[j]=0;
      posInvalids[i] = 0;
      posValids[i] = 0;
      nValids[i] = update_valid_examples(w, m, C, fycache, ex, cached_images, sm, sparm, this_kernels_examples,kernel_info, spl_weight_pos[i], spl_weight_neg[i],&posInvalids[i], &posValids[i]);

      printf("%ldth kernel gives us %d valids %d of which are pos and %d of which are neg\n", i, nValids[i], posValids[i], nValids[i]-posValids[i] );  fflush(stdout);
			printf("Kernels on are ");
			for(j = 0 ; j<m ;j++)
				printf("%d " , this_kernels_examples[j]);
			printf("\n");
			for(j=0; j<m;j++) {
					valid_example_kernels[j][i] = this_kernels_examples[j];
				if (this_kernels_examples[j]) { 
						valid_examples[j] = 1; // since at least one kernel is included for this example, this example is included
				}
			}
			kernel_info[i]=0;
    }

//    for (i=0;i<sm->sizePsi+2;i++)
//    	w[i] = 0.0;

//	for(i=0; i< m; i++)
//	{
//		double total_included=0.0;
//		for(j=0; j< sm->num_kernels; j++)
//		{
//			total_included += valid_example_kernels[i][j];
//		}
//		double percent_included = total_included/sm->num_kernels;
//		ex[i].x.example_cost *= percent_included;
//	}

    if(!sparm->optimizer_type) {
        cutting_plane_algorithm(w, m, MAX_ITER, C, epsilon, fycache, ex, cached_images, sm, sparm, valid_examples, valid_example_kernels);
        } else {
        stochastic_subgradient_descent(w, m, MAX_ITER, C, epsilon, fycache, ex, cached_images, sm, sparm, valid_examples,valid_example_kernels);
     }

	

     double primal_obj;
    primal_obj = current_obj_val(ex, fycache, m, cached_images, sm, sparm, C, prev_valid_examples, prev_valid_example_kernels);

    free(this_kernels_examples);
    free(kernel_info);
	for (i=0;i<m;i++) {
        free(prev_valid_example_kernels[i]);
	}
	free(prev_valid_example_kernels);
	free(prev_valid_examples);
    free(posValids);
    free(posInvalids);
	return(primal_obj);
}


SAMPLE  generate_train_set(SAMPLE alldata, long *perm, int ntrain)
{
  SAMPLE  train;
  train.n = ntrain;
  long i;

  train.examples = (EXAMPLE *) malloc(train.n*sizeof(EXAMPLE));

  for(i = 0; i < train.n; i++)
  {
    train.examples[i] = alldata.examples[perm[i]];
  }

  return train;
}

SAMPLE  generate_validation_set(SAMPLE alldata, long *perm, int ntrain)
{
  SAMPLE  val;
  val.n = alldata.n - ntrain;
  long i;

  val.examples = (EXAMPLE *) malloc(val.n*sizeof(EXAMPLE));

  for(i = 0; i < val.n; i++)
  {
    val.examples[i] = alldata.examples[perm[ntrain+i]];
  }

  return val;
}


int main(int argc, char* argv[]) {

  double *w; /* weight vector */
  int outer_iter;
  long m, i;
  double C, epsilon;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  char trainfile[1024];
  char modelfile[1024];
	char examplesfile[1024];
	char timefile[1024];
	char latentfile[1024];
  int MAX_ITER;
  /* new struct variables */
  SVECTOR **fycache, *diff, *fy;
  EXAMPLE *ex;
	SAMPLE alldata;
  SAMPLE sample;
  SAMPLE val;
  STRUCT_LEARN_PARM sparm;
  STRUCTMODEL sm;
  
  double decrement;
  double primal_obj, last_primal_obj;
  double stop_crit=-1; 
  char itermodelfile[2000];

  /* self-paced learning variables */
  double init_spl_weight;
  double* spl_weight_pos;
  double* spl_weight_neg;
  double spl_factor;
  int *valid_examples;
 

  /* read input parameters */
  my_read_input_parameters(argc, argv, trainfile, modelfile, examplesfile, timefile, latentfile, &learn_parm, &kernel_parm, &sm, &sparm, &init_spl_weight, &spl_factor);

  epsilon = learn_parm.eps*10;
  C = learn_parm.svm_c;
  MAX_ITER = learn_parm.maxiter;

  init_struct_model(get_sample_size(trainfile), (char*)KERNEL_INFO_FILE, &sm);


  /* read in examples */
  alldata = read_struct_examples(trainfile, &sm, &sparm);
  int ntrain = (int) floor(1.0*alldata.n + 0.5); /* no validation set */
    if(ntrain < alldata.n)
    {	
        assert(0);
        srand(time(NULL));
        long *perm = randperm(alldata.n,alldata.n);
        sample = generate_train_set(alldata, perm, ntrain);
        val = generate_validation_set(alldata, perm, ntrain);
        free(perm);
    }
    else
    {
        sample = alldata;
    }
    ex = sample.examples;
    IMAGE_KERNEL_CACHE ** cached_images = init_cached_images(ex,&sm);
    m = sample.n;

    w = create_nvector(sm.sizePsi+1);
    clear_nvector(w, sm.sizePsi+1);
    sm.w = w; /* establish link to w, as long as w does not change pointer */

  /* some training information */
  printf("m = %ld\n", m);
  printf("C: %.8g\n", C);
  printf("spl weight: %.8g\n",init_spl_weight);
  printf("epsilon orig: %.8g\n", epsilon);
  printf("epsilon target: %.8g\n",learn_parm.eps );
  printf("sample.n: %d\n", sample.n); 
  printf("sm.sizePsi: %ld\n", sm.sizePsi); fflush(stdout);
  
  /* impute latent variable for first iteration */
  init_latent_variables(&sample,&learn_parm,&sm,&sparm);

  int k;

  int * all_ones = (int*)calloc(sm.num_kernels, sizeof(int));
  for (k = 0; k < sm.num_kernels; k++) {
    all_ones[k] = 1;
  }

  /* prepare feature vector cache for correct labels with imputed latent variables */
  fycache = (SVECTOR**)malloc(m*sizeof(SVECTOR*));
  for (i=0;i<m;i++) {
    fy = psi(ex[i].x, ex[i].y, ex[i].h, cached_images, all_ones, &sm, &sparm);
    diff = add_list_ss(fy);
    free_svector(fy);
    fy = diff;
    fycache[i] = fy;
  }

 	/* learn initial weight vector using all training examples */
  int j;

  int** allon_example_kernels = (int**)malloc(m*sizeof(int*));
  int** valid_example_kernels = (int**)malloc(m * sizeof(int*));
  for (i = 0; i < m; ++i) {
    allon_example_kernels[i] = (int*)calloc(sm.num_kernels, sizeof(int));
    for(j=0; j<sm.num_kernels; j++) {
        allon_example_kernels[i][j]=1;
    }
    valid_example_kernels[i] = (int*)calloc(sm.num_kernels, sizeof(int));
  }
	valid_examples = (int *) malloc(m*sizeof(int));
    int* allon_examples = (int*) malloc(m*sizeof(int));
    for(i=0; i<m; i++) {
        allon_examples[i]=1;
    } 

	if (init_spl_weight>0.0) {
		printf("INITIALIZATION\n"); fflush(stdout);
		for (i=0;i<m;i++) {
			valid_examples[i] = 1;
			for (j = 0; j < sm.num_kernels; ++j) {
			  valid_example_kernels[i][j] = 1;
			}
		}
		int initIter;
		for (initIter=0;initIter<2;initIter++) { // Rafi: Initial iterations here
		  if(!sparm.optimizer_type) {
		    primal_obj = cutting_plane_algorithm(w, m, MAX_ITER, C, epsilon, fycache, ex, cached_images, &sm, &sparm, valid_examples, valid_example_kernels); // here we use the original epsilon (which is very lenient)
		  } else {
			primal_obj = stochastic_subgradient_descent(w, m, MAX_ITER, C, epsilon, fycache, ex, cached_images, &sm, &sparm, valid_examples,valid_example_kernels);
		  }
  		for (i=0;i<m;i++) {
   	 			LATENT_VAR h_temp = ex[i].h;
      	  ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, cached_images, &sm, &sparm);
          free_latent_var(h_temp);
   		}
	    for (i=0;i<m;i++) {
  	    free_svector(fycache[i]);
	    fy = psi(ex[i].x, ex[i].y, ex[i].h, cached_images, all_ones, &sm, &sparm);
     	 diff = add_list_ss(fy);
     	 free_svector(fy);
     	 fy = diff;
     	 fycache[i] = fy;
    	}
		}
	}
     
  /* outer loop: latent variable imputation */
  outer_iter = 0;
  last_primal_obj = DBL_MAX;
  decrement = 0;

	/* initializations */
	int latent_update = 0;
	FILE	*fexamples = fopen(examplesfile,"w");
	FILE	*ftime = fopen(timefile,"w");
	FILE	*flatent = fopen(latentfile,"w");
	clock_t start = clock();

        spl_weight_pos = (double*)calloc(sm.num_kernels, sizeof(double));
        spl_weight_neg = (double*)calloc(sm.num_kernels, sizeof(double));
        for (k = 0; k < sm.num_kernels; k++) {
          spl_weight_pos[k] = init_spl_weight;
          spl_weight_neg[k] = init_spl_weight;
        }

    //printing some stuff before doing outer loop
    primal_obj = current_obj_val(ex, fycache, m, cached_images, &sm, &sparm, C, allon_examples, allon_example_kernels);
    printf("primal objective (AFTER INITIALIZATION): %f\n", primal_obj);
    double* temp_w = create_nvector(sm.sizePsi+1);
    clear_nvector(temp_w, sm.sizePsi+1);
    double* backup_w = sm.w;
    sm.w = temp_w;
    primal_obj = current_obj_val(ex, fycache, m, cached_images, &sm, &sparm, C, allon_examples, allon_example_kernels);
    printf("primal objective (AFTER INITIALIZATION, W=0): %f\n", primal_obj);
    sm.w = backup_w; //undo any harm done by me
    free(temp_w); 

    //this is the outer loop.   
    while ((outer_iter<2)||((!stop_crit)&&(outer_iter<MAX_OUTER_ITER))) { 
        if(!outer_iter && init_spl_weight) {
            int * valid_kernels = (int*)calloc(sm.num_kernels, sizeof(int));
            if (sparm.multi_kernel_spl) {
                for (k = 0; k < sm.num_kernels; ++k) {
                      valid_kernels[k] = 1;
                      get_init_spl_weight(m, C, fycache, ex, cached_images, valid_kernels, &sm, &sparm,&spl_weight_pos[k],&spl_weight_neg[k]);
                      valid_kernels[k] = 0;
                }
            } else {
                 for (k = 0; k < sm.num_kernels; ++k) {
                      valid_kernels[k] = 1;
                 }
                 get_init_spl_weight(m, C, fycache, ex, cached_images, valid_kernels, &sm, &sparm,&spl_weight_pos[0],&spl_weight_neg[0]);
		         for (k = 1; k < sm.num_kernels; ++k) {
		              spl_weight_pos[k] = spl_weight_pos[0];
		              spl_weight_neg[k] = spl_weight_neg[0];
		         }
		    }  
            free(valid_kernels);
       }
       printf("for negative examples spl weights are %.4f %.4f %.4f %.4f %.4f\n", spl_weight_neg[0], spl_weight_neg[1], spl_weight_neg[2], spl_weight_neg[3], spl_weight_neg[3]);
       printf("for positive examples spl weights are %.4f %.4f %.4f %.4f %.4f\n", spl_weight_pos[0], spl_weight_pos[1], spl_weight_pos[2], spl_weight_pos[3], spl_weight_pos[3]);
    printf("\n\n\nOUTER ITER %d here at epsilon %f \n\n\n", outer_iter, epsilon); 
    /* cutting plane algorithm */

		/* solve biconvex self-paced learning problem */
        primal_obj = alternate_convex_search(w, m, MAX_ITER, C, epsilon, fycache, ex, cached_images, &sm, &sparm, valid_examples, valid_example_kernels, spl_weight_pos, spl_weight_neg);
        int nValid = 0;
        for (i=0;i<m;i++) {
            for(k=0;k<sm.num_kernels;k++) {
                fprintf(fexamples,"%d ",(2*(ex[i].y.label)-1)*valid_example_kernels[i][k]);
            }
            print_latent_var(ex[i].x, ex[i].h,flatent);
            int isValid = 1;
            int this_kernel;
            for(this_kernel=0;this_kernel<sm.num_kernels;this_kernel++){
                if(!valid_example_kernels[i][this_kernel])
                {
                    isValid=0;
                }
            }
            nValid+=isValid;
        }
        fprintf(fexamples,"\n"); fflush(fexamples);
        fprintf(flatent,"\n"); fflush(flatent);
        clock_t finish = clock();
        fprintf(ftime,"%f %f\n",primal_obj,(((double)(finish-start))/CLOCKS_PER_SEC)); fflush(ftime);
        
        /* compute decrement in objective in this outer iteration */
        decrement = last_primal_obj - primal_obj;
        last_primal_obj = primal_obj;
        printf("primal objective (THIS IS THE MONEY SHOT): %.4f\n", primal_obj);
        if (outer_iter) {
            printf("decrement (outer iter): %.4f\n", decrement); fflush(stdout);
        }
        else {
            printf("decrement (outer iter): N/A\n"); fflush(stdout);
        }
       
        /* additional stopping criteria */
             
        /* impute latent variable using updated weight vector */
        latent_update=0;
        if(nValid) {
            for (i=0;i<m;i++) {
                LATENT_VAR h_temp = ex[i].h;
                ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, cached_images, &sm, &sparm);
                free_latent_var(h_temp);
            }
        }
        stop_crit = 1;
        if(decrement>C*epsilon)
        {
            printf("NOT STOPPING BECAUSE OF DESCENT\n");
            stop_crit = 0;
        }
        
        if(nValid < m)
        {
            printf("NOT STOPPING BECAUSE OF NVALIDS\n");
            stop_crit = 0;
        }

        if(epsilon>learn_parm.eps)
        {
            printf("NOT STOPPING BECAUSE OF EPSILON\n");
            epsilon = MAX(learn_parm.eps, epsilon/3);
            stop_crit = 0;
        }
        /* re-compute feature vector cache */
        for (i=0;i<m;i++) {
          free_svector(fycache[i]);
          fy = psi(ex[i].x, ex[i].y, ex[i].h, cached_images, all_ones, &sm, &sparm);
          diff = add_list_ss(fy);
          free_svector(fy);
          fy = diff;
          fycache[i] = fy;
        }
		printf("computing new objective value\n"); fflush(stdout);
        primal_obj = current_obj_val(ex, fycache, m, cached_images, &sm, &sparm, C, allon_examples, allon_example_kernels);
        printf("primal object (AFTER IMPUTATION): %f\n", primal_obj);
		sprintf(itermodelfile,"%s.%04d",modelfile,outer_iter);
		write_struct_model(itermodelfile, &sm, &sparm);

        outer_iter++;
        for(i=0;i<sm.num_kernels;i++) {
            if(init_spl_weight)
            {
                spl_weight_pos[i] += spl_factor;
                spl_weight_neg[i] += spl_factor;
                spl_weight_pos[i] = (spl_weight_pos[i]>1) ? 1 : spl_weight_pos[i];
                spl_weight_neg[i] = (spl_weight_neg[i]>1) ? 1:spl_weight_neg[i];
            }
        }
  } // end outer loop
  fclose(fexamples);
  fclose(ftime);
  fclose(flatent);
  

  /* write structural model */
  write_struct_model(modelfile, &sm, &sparm);
  // skip testing for the moment  

  /* free memory */
  free_cached_images(cached_images, &sm);
  free_struct_sample(alldata);
	if(ntrain < alldata.n)
	{
		free(sample.examples);
		free(val.examples);
	}
  free_struct_model(sm, &sparm);
  for(i=0;i<m;i++) {
    free_svector(fycache[i]);
  }
  free(fycache);

  free(all_ones);

	free(valid_examples);
   
  return(0); 
  
}



void my_read_input_parameters(int argc, char *argv[], char *trainfile, char* modelfile, char *examplesfile, char *timefile, char *latentfile,
			      LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm, STRUCTMODEL * sm, STRUCT_LEARN_PARM *struct_parm,
						double *init_spl_weight, double *spl_factor) {
  
  long i;
	char filestub[1024];

  /* set default */
  struct_parm->pos_neg_cost_ratio = 1.0;
  learn_parm->maxiter=20000;
  learn_parm->svm_maxqpsize=100;
  learn_parm->svm_c=100.0;
  learn_parm->eps=0.01;
  learn_parm->biased_hyperplane=12345; /* store random seed */
  learn_parm->remove_inconsistent=10; 
  kernel_parm->kernel_type=0;
  kernel_parm->rbf_gamma=0.05;
  kernel_parm->coef_lin=1;
  kernel_parm->coef_const=1;
  kernel_parm->poly_degree=3;
	/* default: no self-paced learning */
	*init_spl_weight = 0.0;
	*spl_factor = 1.3;
	struct_parm->optimizer_type = 1; /* default: cutting plane, change to 1 for stochastic subgradient descent*/
	struct_parm->init_valid_fraction = 0.5;
	struct_parm->margin_type = 0;

  struct_parm->custom_argc=0;
  struct_parm->multi_kernel_spl = 0;

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) {
    case 'c': i++; learn_parm->svm_c=atof(argv[i]); break;
    case 'e': i++; learn_parm->eps=atof(argv[i]); break;
    case 's': i++; learn_parm->svm_maxqpsize=atol(argv[i]); break; 
    case 'g': i++; kernel_parm->rbf_gamma=atof(argv[i]); break;
    case 'd': i++; kernel_parm->poly_degree=atol(argv[i]); break;
    case 'r': i++; learn_parm->biased_hyperplane=atol(argv[i]); break; 
    case 't': i++; kernel_parm->kernel_type=atol(argv[i]); break;
    case 'n': i++; learn_parm->maxiter=atol(argv[i]); break;
    case 'p': i++; learn_parm->remove_inconsistent=atol(argv[i]); break; 
    case 'z': i++; struct_parm->multi_kernel_spl = atol(argv[i]); break;
		case 'k': i++; *init_spl_weight = atof(argv[i]); break;
		case 'm': i++; *spl_factor = atof(argv[i]); break;
		case 'o': i++; struct_parm->optimizer_type = atoi(argv[i]); break;
		case 'f': i++; struct_parm->init_valid_fraction = atof(argv[i]); break;
    case '-': strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);i++; strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);break; 
    default: printf("\nUnrecognized option %s!\n\n",argv[i]);
      exit(0);
    }
  }
  assert(*init_spl_weight > 0.0 || !struct_parm->multi_kernel_spl);

  *init_spl_weight = (*init_spl_weight)/learn_parm->svm_c;

  if(i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    my_wait_any_key();
    exit(0);
  }

  strcpy (trainfile, argv[i]);

  if((i+1)<argc) {
    strcpy (modelfile, argv[i+1]);
  }
	else {
		strcpy (modelfile, "lssvm.model");
	}

	if((i+2)<argc) {
		strcpy (filestub, argv[i+2]);
	}
	else {
		strcpy (filestub, "lssvm");
	}

	sprintf(examplesfile,"%s.examples",filestub);
	sprintf(timefile,"%s.time",filestub);
	sprintf(latentfile,"%s.latent",filestub);

	/* self-paced learning weight should be non-negative */
	if(*init_spl_weight < 0.0)
		*init_spl_weight = 0.0;
	/* self-paced learning factor should be greater than 1.0 */

  parse_struct_parameters(struct_parm);
}


void my_wait_any_key()
{
  printf("\n(more)\n");
  (void)getc(stdin);
}

int resize_cleanup(int size_active, int **ptr_idle, double **ptr_alpha, double** ptr_alpha_lb, double **ptr_delta, double ** ptr_delta_plus_A_wlast, DOC ***ptr_dXc, 
		double ***ptr_G, int *mv_iter) {
  int i,j, new_size_active;
  long k;

  int *idle=*ptr_idle;
  double *alpha=*ptr_alpha;
  double *alpha_lb=*ptr_alpha_lb;
  double *delta=*ptr_delta;
	double *delta_plus_A_wlast = *ptr_delta_plus_A_wlast;
	DOC	**dXc = *ptr_dXc;
	double **G = *ptr_G;
	int new_mv_iter = -1; //I'm assuming that this always gets set later (it used to be uninitialized)

  i=0;
  while ((i<size_active)&&(idle[i]<IDLE_ITER)) i++;
  j=i;
  while((j<size_active)&&(idle[j]>=IDLE_ITER)) j++;

  while (j<size_active) {
    /* copying */
    alpha[i] = alpha[j];
    alpha_lb[i] = alpha_lb[j];
    delta[i] = delta[j];
    delta_plus_A_wlast[i] = delta_plus_A_wlast[j];
		free(G[i]);
		G[i] = G[j];
		G[j] = NULL;
    free_example(dXc[i],1);
    dXc[i] = dXc[j];
    dXc[j] = NULL;
		if(j == *mv_iter) {
			new_mv_iter = i;
		}
    i++;
    j++;
    while((j<size_active)&&(idle[j]>=IDLE_ITER)) j++;
  }


  for (k=i;k<size_active;k++) {
		if (G[k]!=NULL) free(G[k]);
    if (dXc[k]!=NULL) free_example(dXc[k],1);
  }
	*mv_iter = new_mv_iter;
  new_size_active = i;
  alpha = (double*)realloc(alpha, sizeof(double)*new_size_active);
  alpha_lb = (double*)realloc(alpha_lb, sizeof(double)*new_size_active);
  delta = (double*)realloc(delta, sizeof(double)*new_size_active);
	delta_plus_A_wlast = (double*)realloc(delta_plus_A_wlast, sizeof(double) * new_size_active);
	G = (double **) realloc(G, sizeof(double *)*new_size_active);
  dXc = (DOC**)realloc(dXc, sizeof(DOC*)*new_size_active);
  assert(dXc!=NULL);

  /* resize idle */
  i=0;
  while ((i<size_active)&&(idle[i]<IDLE_ITER)) i++;
  j=i;
  while((j<size_active)&&(idle[j]>=IDLE_ITER)) j++;

  while (j<size_active) {
    idle[i] = idle[j];
		for (k=0;k<new_size_active;k++) {
			G[k][i] = G[k][j];
		}
    i++;
    j++;
    while((j<size_active)&&(idle[j]>=IDLE_ITER)) j++;
  }  
  idle = (int*)realloc(idle, sizeof(int)*new_size_active);
	for (k=0;k<new_size_active;k++) {
		G[k] = (double*)realloc(G[k], sizeof(double)*new_size_active);
	}

  *ptr_idle = idle;
  *ptr_alpha = alpha;
  *ptr_alpha_lb = alpha_lb;
  *ptr_delta = delta;
	*ptr_delta_plus_A_wlast = delta_plus_A_wlast;
	*ptr_G = G;
  *ptr_dXc = dXc;

  return(new_size_active);
}

void approximate_to_psd(double **G, int size_active, double eps)
{
	int i,j,k;
	double *copy_G = (double*) malloc(size_active*size_active*sizeof(double));
	double *eig_vec =(double*)malloc(size_active*size_active*sizeof(double));
	double *eig_val =(double*) malloc(size_active*sizeof(double));

	for(i = 0; i < size_active; i++)
		for(j = 0; j < size_active; j++)
			copy_G[i*size_active+j] = G[i][j];

	Jacobi_Cyclic_Method(eig_val,eig_vec,copy_G,size_active);

	for(i = 0; i < size_active; i++)
		for(j = 0; j < size_active; j++)
		{
			copy_G[i*size_active+j] = MAX(eig_val[i],eps)*eig_vec[j*size_active+i];
		}

	for(i = 0; i < size_active; i++)
		for(j = 0; j < size_active; j++)
		{
			G[i][j] = 0.0;
			for(k = 0; k < size_active; k++)
			{
				G[i][j] += eig_vec[i*size_active+k]*copy_G[k*size_active+j];
			}
		}

	free(copy_G);
	free(eig_vec);
	free(eig_val);
}

void Jacobi_Cyclic_Method(double eigenvalues[], double *eigenvectors, double *A, int n)
{
   int i, j, k, m;
   double *pAk, *pAm, *p_r, *p_e;
   double threshold_norm;
   double threshold;
   double tan_phi, sin_phi, cos_phi, tan2_phi, sin2_phi, cos2_phi;
   double sin_2phi, cos_2phi, cot_2phi;
   double dum1;
   double dum2;
   double dum3;
   double max;

                  // Take care of trivial cases

   if ( n < 1) return;
   if ( n == 1) {
      eigenvalues[0] = *A;
      *eigenvectors = 1.0;
      return;
   }

          // Initialize the eigenvalues to the identity matrix.

   for (p_e = eigenvectors, i = 0; i < n; i++)
      for (j = 0; j < n; p_e++, j++)
         if (i == j) *p_e = 1.0; else *p_e = 0.0;
  
            // Calculate the threshold and threshold_norm.
 
   for (threshold = 0.0, pAk = A, i = 0; i < ( n - 1 ); pAk += n, i++) 
      for (j = i + 1; j < n; j++) threshold += *(pAk + j) * *(pAk + j);
   threshold = sqrt(threshold + threshold);
   threshold_norm = threshold * DBL_EPSILON;
   max = threshold + 1.0;
   while (threshold > threshold_norm) {
      threshold /= 10.0;
      if (max < threshold) continue;
      max = 0.0;
      for (pAk = A, k = 0; k < (n-1); pAk += n, k++) {
         for (pAm = pAk + n, m = k + 1; m < n; pAm += n, m++) {
            if ( fabs(*(pAk + m)) < threshold ) continue;

                 // Calculate the sin and cos of the rotation angle which
                 // annihilates A[k][m].

            cot_2phi = 0.5 * ( *(pAk + k) - *(pAm + m) ) / *(pAk + m);
            dum1 = sqrt( cot_2phi * cot_2phi + 1.0);
            if (cot_2phi < 0.0) dum1 = -dum1;
            tan_phi = -cot_2phi + dum1;
            tan2_phi = tan_phi * tan_phi;
            sin2_phi = tan2_phi / (1.0 + tan2_phi);
            cos2_phi = 1.0 - sin2_phi;
            sin_phi = sqrt(sin2_phi);
            if (tan_phi < 0.0) sin_phi = - sin_phi;
            cos_phi = sqrt(cos2_phi); 
            sin_2phi = 2.0 * sin_phi * cos_phi;
            cos_2phi = cos2_phi - sin2_phi;

                     // Rotate columns k and m for both the matrix A 
                     //     and the matrix of eigenvectors.

            p_r = A;
            dum1 = *(pAk + k);
            dum2 = *(pAm + m);
            dum3 = *(pAk + m);
            *(pAk + k) = dum1 * cos2_phi + dum2 * sin2_phi + dum3 * sin_2phi;
            *(pAm + m) = dum1 * sin2_phi + dum2 * cos2_phi - dum3 * sin_2phi;
            *(pAk + m) = 0.0;
            *(pAm + k) = 0.0;
            for (i = 0; i < n; p_r += n, i++) {
               if ( (i == k) || (i == m) ) continue;
               if ( i < k ) dum1 = *(p_r + k); else dum1 = *(pAk + i);
               if ( i < m ) dum2 = *(p_r + m); else dum2 = *(pAm + i);
               dum3 = dum1 * cos_phi + dum2 * sin_phi;
               if ( i < k ) *(p_r + k) = dum3; else *(pAk + i) = dum3;
               dum3 = - dum1 * sin_phi + dum2 * cos_phi;
               if ( i < m ) *(p_r + m) = dum3; else *(pAm + i) = dum3;
            }
            for (p_e = eigenvectors, i = 0; i < n; p_e += n, i++) {
               dum1 = *(p_e + k);
               dum2 = *(p_e + m);
               *(p_e + k) = dum1 * cos_phi + dum2 * sin_phi;
               *(p_e + m) = - dum1 * sin_phi + dum2 * cos_phi;
            }
         }
         for (i = 0; i < n; i++)
            if ( i == k ) continue;
            else if ( max < fabs(*(pAk + i))) max = fabs(*(pAk + i));
      }
   }
   for (pAk = A, k = 0; k < n; pAk += n, k++) eigenvalues[k] = *(pAk + k); 
}
