// classes example
#ifndef _KERNEL_H
#define _KERNEL_H

#include <iostream>
#include <assert.h>

class kernel_obj {
    
    public:
        int total_length; // THIS IS LENGTH OF kernel_weights VECTOR.
        int num_kernels;
        int is_spm;
        int* kernel_full_lengths; //counting BoW/SPM duplication AND factor of 2.
        int* kernel_lengths;
        int* kernel_starts; // inclusive
        int* kernel_ends; //you own this one, inclusive
        double* kernel_weights;
     // Format : init_kernel_weights[0] = 0, init_kernel_weights[1]=1 or LEARNED BIAS.
        void initialize(int init_num_kernels, int* init_kernel_lengths, double* init_kernel_weights, int init_is_spm)
        {
            assert(!kernel_lengths); // initialize once or feel my wrath
            assert(!kernel_weights);

    
            is_spm = init_is_spm;
            int multiplier = is_spm ? 5 :1;

            num_kernels= init_num_kernels;
            kernel_lengths = init_kernel_lengths;
            total_length = 2;
            kernel_starts = (int*)malloc(sizeof(int)*num_kernels);
            kernel_ends = (int*)malloc(sizeof(int)*num_kernels);
            kernel_full_lengths =(int*)malloc(sizeof(int)*num_kernels);
            for(int i = 0 ; i < num_kernels ; i++)
            {
                kernel_starts[i] = total_length;
                kernel_full_lengths[i] = 2* multiplier*kernel_lengths[i];
                kernel_ends[i] = total_length + (kernel_full_lengths[i]) -1;  
                total_length += kernel_full_lengths[i];
            }

            if(init_kernel_weights)
                kernel_weights= init_kernel_weights;
            else
            {
                kernel_weights = (double*)malloc(multiplier*total_length* sizeof(double));
                for(int i = 0 ; i<multiplier*total_length;i++)
                    kernel_weights[i]=0;
            }
        }
        void cleanup()
        {
            free(kernel_weights);
            free(kernel_starts);
            free(kernel_ends);
            free(kernel_full_lengths);
        } 
        kernel_obj()
        {
            total_length=-1;
            num_kernels = -1;
            kernel_lengths=NULL;
            kernel_weights=NULL;
        }

        void set(int kernel_num, int index, int box_choice, double weight)
        {
            if(! is_spm)
            {
                assert((kernel_starts[kernel_num]+index) <= kernel_ends[kernel_num]);
                kernel_weights[kernel_starts[kernel_num]+index] = weight;
            }
            else
            {
                assert(index>=0 && index < kernel_lengths[kernel_num]);
                int point = kernel_starts[kernel_num] + index + (box_choice*(kernel_lengths[kernel_num]));
                assert(point <= kernel_ends[kernel_num]);
                kernel_weights[point] = weight;
            }
        }

        double  get(int kernel_num, int index, int box_choice)
        {
            if(! is_spm)
            {
                assert((kernel_starts[kernel_num]+index) <= kernel_ends[kernel_num]);
                return  kernel_weights[kernel_starts[kernel_num]+index];
            }
            else
            {
                assert(index>=0 && index < kernel_lengths[kernel_num]);
                int point = kernel_starts[kernel_num] + index + (box_choice*(kernel_lengths[kernel_num]));
                assert(point <= kernel_ends[kernel_num]);
                return kernel_weights[point];
            }
        }

        //YOU DON'T OWN THIS SO DON'T FREE IT.
        double* get_vec()
        {
            return kernel_weights;
        }


    private:
};


#endif

