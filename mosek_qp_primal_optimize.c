/* 
   Copyright: Copyright (c) 1998-2006 MOSEK ApS, Denmark. All rights is reserved.

   File:      qo1.c

   Purpose:   Demonstrate how to solve a quadratic
              optimization problem using the MOSEK API.
 */
#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mosek.h" /* Include the MOSEK definition file. */

static void MSKAPI printstr(void *handle,
                            char str[])
{
//  printf("%s",str);
} /* printstr */


//int rafi_solver(int size_w, int num_constraints, double C, double** cons, double* margins)
int mosek_qp_primal_optimize(double** cons, double* margins, double* objective,double* w, double C, double size_w, int num_constraints)
{
  int NUMVAR = size_w+1;
  int NUMCON = num_constraints;
  MSKenv_t   env;
  MSKrescodee r;
  MSKtask_t  task;
  r = MSK_makeenv(&env,NULL,NULL,NULL,NULL);
  if(r==MSK_RES_OK) 
      MSK_linkfunctoenvstream(env,
                            MSK_STREAM_LOG,
                            NULL,
                            printstr);

  if(r==MSK_RES_OK) r = MSK_initenv(env);
  if(r==MSK_RES_OK) r = MSK_maketask(env,0,NUMVAR,&task);
  if(r==MSK_RES_OK) r = MSK_linkfunctotaskstream(task,MSK_STREAM_LOG,NULL,printstr);
  if(r == MSK_RES_OK)
  {
    r = MSK_append(task,MSK_ACC_CON,NUMCON);
    r = MSK_append(task,MSK_ACC_VAR,NUMVAR);
  }

   if(r==MSK_RES_OK) r = MSK_putcj(task,0,C);

     if(r == MSK_RES_OK)
          r = MSK_putbound(task,
                           MSK_ACC_VAR, /* Put bounds on variables.*/
                           0,           /* Index of variable.*/
                           MSK_BK_LO,      /* Bound key.*/
                           0,      /* Numerical value of lower bound.*/
                           +MSK_INFINITY);     /* Numerical value of upper bound.*/
    
    for(int j = 1 ; j < NUMVAR; j++)
    {
        if(r == MSK_RES_OK)
          r = MSK_putbound(task,
                           MSK_ACC_VAR, /* Put bounds on variables.*/
                           j,           /* Index of variable.*/
                           MSK_BK_FR,      /* Bound key.*/
                           -MSK_INFINITY,      /* Numerical value of lower bound.*/
                           MSK_INFINITY);     /* Numerical value of upper bound.*/
    }

    MSKidxt indices[NUMVAR];
    for(int i = 0 ; i < NUMVAR; i++)
        indices[i] = i;

    for(int i = 0 ; i< NUMCON ; i++)
    {
        r = MSK_putbound(task, MSK_ACC_CON ,    /*  Put bounds  on  constraints .*/
                            i,  /* Index    of  constraint .*/ 
                            MSK_BK_LO,  /* Bound    key .*/ 
                            margins[i], /* Numerical value of lower bound .*/ 
                            +MSK_INFINITY);  /* Numerical value of upper bound .*/

        r = MSK_putavec(task, MSK_ACC_CON , /*  Input row of A.*/
                            i,  /* Row  index .*/ 
                            NUMVAR, /* Number of non-zeros in row i.*/ 
                            indices,  /* Pointer to column indexes of row i.*/ 
                            cons[i]);    /* Pointer to Values of row i.*/
    }

    
    

    if ( r==MSK_RES_OK )
      {
        /*
         * The lower triangular part of the Q
         * matrix in the objective is specified.
         */
        MSKidxt* qsubi = (MSKidxt*)malloc(sizeof(MSKidxt)*(size_w));
        MSKidxt* qsubj = (MSKidxt*)malloc(sizeof(MSKidxt)*(size_w));
        double* qval = (double*)malloc(sizeof(double)*(size_w));

        for(int i = 1; i <= size_w; i++)
        {
            qsubi[i-1]=i;
            qsubj[i-1]=i;
            qval[i-1]=1;
        }

        /* Input the Q for the objective. */

        r = MSK_putqobj(task,size_w,qsubi,qsubj,qval);
      }

      MSK_putdouparam(task, MSK_DPAR_INTPNT_TOL_REL_GAP, 1E-14);
        
      if ( r==MSK_RES_OK )
        r = MSK_optimize(task);


      if ( r==MSK_RES_OK )
      {

        MSK_getsolutionslice(task,
                             MSK_SOL_ITR,
                             MSK_SOL_ITEM_XX,
                             1,
                             NUMVAR,
                             (MSKrealt*)w);

      }
  MSK_getprimalobj(task, MSK_SOL_ITR, objective);
  MSK_deletetask(&task);
  
  MSK_deleteenv(&env);


  return ( r );
}
/*int main()
{
    int num_vars = 5000;
    int num_constraints = 20;
    double C = 100;
    int count = 0;
    
    double* margins = (double*) malloc(sizeof(double)*num_constraints);
    double** cons = (double**) malloc(sizeof(double*) * num_constraints);

    srand(12);
    while(count < 100) {
        count++;
	    for(int i = 0 ; i < num_constraints ; i++)
	    {
	        margins[i] = 1;
	        cons[i] = (double*) malloc(sizeof(double)*(1+num_vars));
	        for(int j = 0 ; j < num_vars+1; j++)
	        {
	            cons[i][j] = -0.5+rand()/(RAND_MAX+1.0);
	            if(j==0)
	                cons[i][0]=1;
	        } 
	    }
        std::clock_t start = std::clock();
        printf("pre solve\n");
        rafi_solver(num_vars, num_constraints, C, cons, margins);
        printf("post solve\n");
        std::clock_t post = std::clock();
        printf("Took %f\n", ( ( post- start ) / (double)CLOCKS_PER_SEC ));
        getchar();
    }

}*/

