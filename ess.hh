/********************************************************
 *                                                      *
 * Efficient Subwindow Search (ESS) implemention in C++ *
 * entry point and commmand line interface              *
 *                                                      *
 *  Performs a branch-and-bound search for an arbitrary *
 *  quality function, provided a routine to bound it,   *
 *  see  C. H. Lampert, M. B. Blaschko and T. Hofmann:  *
 *       Beyond Sliding Windows: Object Localization    *
 *       by Efficient Subwindow Search, CVPR (2008)     *
 *                                                      *
 *   Copyright 2006-2008 Christoph Lampert              *
 *   Contact: <mail@christoph-lampert.org>              *
 *                                                      *
 *  Licensed under the Apache License, Version 2.0 (the *
 *  "License"); you may not use this file except in     *
 *  compliance with the License. You may obtain a copy  *
 *  of the License at                                   *
 *                                                      *
 *     http://www.apache.org/licenses/LICENSE-2.0       *
 *                                                      *
 *  Unless required by applicable law or agreed to in   *
 *  writing, software distributed under the License is  *
 *  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES *
 *  OR CONDITIONS OF ANY KIND, either express or        *
 *  implied. See the License for the specific language  *
 *  governing permissions and limitations under the     *
 *  License.                                            *
 *                                                      *
 ********************************************************/

#ifndef _ESS_H
#define _ESS_H

#include <stdio.h>
#include <limits>
#include <string>
#include <queue>

// structure holding a single box
typedef struct { 
        int left;
        int top;
        int right;
        int bottom;
        double score;
} Box;

// structure to hold a set of boxes ( = a state during search)
class sstate {
  public:
    float upper; 
    short only[4];   // there can be millions of sstates, better save some memory

    // construct empty
    sstate() { }
    
    // construct as full search space
    sstate(int argwidth, int argheight) {
        upper = std::numeric_limits<float>::max();
        only[0] = 1;
        only[1] = 1;
        only[2] = argwidth;
        only[3] = argheight;
    }

    std::string tostring() const {
        char cstring[80];
        sprintf(cstring, "low < %d %d %d %d > high < %d %d %d %d >\n", 
                only[0],only[1],only[2],only[3],only[0],only[1],only[2],only[3]);
        return std::string(cstring);
    }

    // calculate argmax_i( only[i]-only[i] ), or -1 if high==low
    int maxindex() const { 
        int splitindex=-1;
        int maxwidth=0;
        for (unsigned int i=0;i<4;i++) { 
            int interval_width = only[i] - only[i];
            if (interval_width > maxwidth) {
                splitindex=i;
                maxwidth = interval_width;
            }
        }
        return splitindex;
    }

    bool islegal() const {
        return ((only[0] <= only[2]) && (only[1] <= only[3]));
    }

    // "less" is needed to compare states in stl_priority queue
    bool less(const sstate* other) const {
        return upper < other->upper;
    }
};


// helper routine for comparing elements in priority queue
class sstate_comparisson {
  public:
    bool operator() (const sstate* lhs, const sstate* rhs) const {
        return lhs->less(rhs);
    }
};

// data structure for priority queue
typedef std::priority_queue<const sstate*, 
                            std::vector<const sstate*>, 
                            sstate_comparisson> sstate_heap;


extern "C" {
Box pyramid_search(int argnumpoints, int argwidth, int argheight,
                   double* argxpos, double* argypos, double* argclst,
                   int argnumclusters, int argnumlevels, double* argweight,
									 double maxiterations, int& solvedExactly, int factor);
}

#endif
