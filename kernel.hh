// classes example
#ifndef _KERNEL_H
#define _KERNEL_H

#include <iostream>
#include <assert.h>
#include "svm_light/svm_common.h"

class kernel_obj {
    
    public:
        int total_length; // THIS IS LENGTH OF kernel_weights VECTOR.
	int section_length;
	int num_sections;
        int num_kernels;
        int is_spm;
        int* kernel_full_lengths; //counting BoW/SPM duplication AND factor of 2.
        int* kernel_lengths;
        int* kernel_starts; // inclusive
        int* kernel_ends; //you own this one, inclusive
        double* kernel_weights;
     // Format : init_kernel_weights[0] = 0, init_kernel_weights[1]=1 or LEARNED BIAS.
        void initialize(int init_num_kernels, int* init_kernel_lengths, int init_num_sections, double* init_kernel_weights, int init_is_spm)
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

                num_sections = init_num_sections;
                section_length = total_length - 1;
                total_length = num_sections * section_length + 1;

            if(init_kernel_weights) {
		assert(0);
                kernel_weights= init_kernel_weights;
	    } else {
                kernel_weights = (double*)malloc(total_length * sizeof(double));
                for(int i = 0 ; i < total_length; i++) {
                    kernel_weights[i]=0;
		}
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
		num_sections = -1;
		section_length = -1;
            num_kernels = -1;
            kernel_lengths=NULL;
            kernel_weights=NULL;
        }
      
        int  get_index(int kernel_num, int index, int box_choice, bool in_bb, int section_num)
        {
            int bb_multiplier = in_bb ? 0 : 1;
            int point_index;
            if (! is_spm) 
                point_index = kernel_starts[kernel_num]+index + (kernel_lengths[kernel_num]*bb_multiplier);
            else
                point_index = kernel_starts[kernel_num] + index + (box_choice*(kernel_lengths[kernel_num])) + 5*(kernel_lengths[kernel_num]*bb_multiplier);

	    point_index += section_num * section_length;

            return point_index;
           
        }

        void set(int kernel_num, int index, int box_choice, bool in_bb, int section_num, double weight)
        {
            kernel_weights[get_index(kernel_num, index, box_choice, in_bb, section_num)] = weight;
        }

        double  get(int kernel_num, int index, int box_choice, bool in_bb, int section_num)
        {
            return kernel_weights[get_index(kernel_num, index, box_choice, in_bb, section_num)];
        }

        //YOU DON'T OWN THIS SO DON'T FREE IT.
        double* get_vec()
        {
            return kernel_weights;
        }

	SVECTOR * get_svec(int section_num)
	{
		//The section starts at section_num * section_length + 1, but we need to include the extra 0 at the beginnning
		SVECTOR * fvec = create_svector_n(&(kernel_weights[section_num * section_length]), section_length, "", 1.0);
		WORD * word = fvec->words;
		while (word->wnum) {
			word->wnum += section_num * section_length;
			++word;
		}
		return fvec;
	}

    private:
};


#endif

