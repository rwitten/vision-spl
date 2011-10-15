// classes example
#ifndef _KERNEL_H
#define _KERNEL_H

#include <iostream>
#include <assert.h>

class kernel_obj {
    int total_length; // THIS IS LENGTH OF kernel_weights VECTOR.
    int num_kernels;
    int* kernel_lengths;
    int* kernel_starts; // inclusive
    int* kernel_ends; //you own this one, inclusive
    double* kernel_weights;

    public:
        // Format : init_kernel_weights[0] = 0, init_kernel_weights[1]=1 or LEARNED BIAS.
        void initialize(int init_num_kernels, int* init_kernel_lengths, double* init_kernel_weights)
        {
            assert(!kernel_lengths); // initialize once or feel my wrath
            assert(!kernel_weights);

            num_kernels= init_num_kernels;
            kernel_lengths = init_kernel_lengths;
            kernel_weights = init_kernel_weights;
            total_length = 2;
            kernel_starts = (int*)malloc(sizeof(int)*num_kernels);
            kernel_ends = (int*)malloc(sizeof(int)*num_kernels);
            for(int i = 0 ; i < num_kernels ; i++)
            {
                kernel_starts[i] = total_length;
                kernel_ends[i] = total_length + kernel_lengths[i] -1;  
                total_length += kernel_lengths[i];
            }
            if(init_kernel_weights)
                kernel_weights= init_kernel_weights;
            else
                kernel_weights = (double*)calloc(total_length, sizeof(double));
        }
    
        kernel_obj()
        {
            total_length=-1;
            num_kernels = -1;
            kernel_lengths=NULL;
            kernel_weights=NULL;
        }

        void set(int kernel_num, int index, double weight)
        {
            assert((kernel_starts[kernel_num]+index) <= kernel_ends[kernel_num]);
            kernel_weights[kernel_starts[kernel_num]+index] = weight;
        }

        double  get(int kernel_num, int index)
        {
            assert((kernel_starts[kernel_num]+index) <= kernel_ends[kernel_num]);
            return  kernel_weights[kernel_starts[kernel_num]+index];
        }

        //YOU DON'T OWN THIS SO DON'T FREE IT.
        double* get_vec()
        {
            return kernel_weights;
        }
    private:
};


#endif

