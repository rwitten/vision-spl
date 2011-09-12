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

#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>

#include "ess.hh"
#include "quality_pyramid.hh"

#define MAXDATAPOINTS 100000 // ad hoc limits
#define MAXWIDTH 8192
#define MAXHEIGHT 8192
#define MAXCLUSTERS 100000

static int maxresults = 1;
static int numlevels = 1;
static int verbose = 0;

// Here we chose the class to calculate quality bounds for us.
// It has to have at least the interface of the QualityFunction class.
//
// We use 'PyramidQualityFunction', because it's flexible.
// We use 'BoxQualityFunction' is a little easier to set up.


// central routine during branch-and-bound search:
// 1) extract the most promising candidate region 
// 2) split it, if necessary 
// 3) calculate upper bounds for the parts
// 4) re-insert the parts


//find the biggest number of the form c*base + 1 less than or equal to  input
inline int round_down(int input, int base)
{
	if(base* ((int)(input/base))+1 <= input)
		return base* ((int)(input/base))+1;
	else
		return base* ((int)(input/base))+1-base;
}

inline int round_strictly_up(int input, int base)
{
	if(base* ((int)(input/base))+1 > input)
		return base* ((int)(input/base))+1;
	else
		return base* ((int)(input/base))+1+base;
}

static int extract_split_and_insert(sstate_heap *pH,PyramidQualityFunction quality_bound,int factor) {

    // step 1) find the most promising candidate region 
    const sstate* curstate = pH->top();
//		printf("best upper bound is %f\n", curstate->upper);
    // step 2a) check if the stop criterion is reached
    const int splitindex = curstate->maxindex();

    if (splitindex < 0)
        return -1;    // no more splits => convergence
//		printf("spliting on %d %d\n", curstate->only[splitindex], curstate->only[splitindex]);
    // step 2b) otherwise, create two new states as copies of the old, except for splitindex
    pH->pop();
    sstate* newstate0 = new sstate(*curstate);
    newstate0->only[splitindex] = (curstate->only[splitindex] + curstate->only[splitindex])>>1;
		//printf("We round down %d to %d\n", newstate0->only[splitindex], round_down(newstate0->only[splitindex],factor));
    newstate0->only[splitindex] = round_down(newstate0->only[splitindex],factor);

    sstate* newstate1 = new sstate(*curstate);
    newstate1->only[splitindex] = (curstate->only[splitindex] + curstate->only[splitindex]+1)>>1;
    newstate1->only[splitindex] = round_strictly_up(newstate1->only[splitindex],factor);
		//printf("We round up %d to %d\n", newstate1->only[splitindex], round_up(newstate1->only[splitindex],factor));

//		printf("We split %d [%d, %d] into [%d, %d] and [%d, %d]\n", splitindex,curstate->only[splitindex], curstate->only[splitindex],newstate0->only[splitindex],newstate0->only[splitindex], newstate1->only[splitindex], newstate1->only[splitindex]);
    // the old state isn't needed anymore
    delete curstate; curstate=NULL;
    
    // step 3&4) calculate upper bounds for the parts and reinject them 
    if ( newstate0->islegal() ) {
        newstate0->upper = quality_bound.upper_bound(newstate0);
        pH->push(newstate0);
    }
		else
		{
			free(newstate0);
		}

    if ( newstate1->islegal() ) {
        newstate1->upper = quality_bound.upper_bound(newstate1);
        pH->push(newstate1);
    }
		else
		{
			free(newstate1);
		}
    
    // no error, but also no convergence, yet
    return 0;
}

extern "C" {

// main entry site for efficient subwindow search.
// performs preprocessing and then branch-and-bound
// We make it "extern C", so it's easier to call e.g. from Python
//
// INPUT: int argnumpoints,     : number of data points
//        int width, height     : width and height of image
//        double* argxpos,      : x-coordinate of every point
//        double* argypos,      : y-coordinate of every point
//        double* argclst       : cluster ID of each point (starts at 0)
//        int argnumclusters,   : number of clusterIDs 
//        int argnumlevels,     : number of levels in the pyramid 
//        double* weightsdata   : vector of cluster weights
// OUTPUT: Box outputBox        : box in [left,top,right,bottom,score] format

Box pyramid_search(int argnumpoints, int argwidth, int argheight, 
                   double* argxpos, double* argypos, double* argclst,
                   int argnumclusters, int argnumlevels, double* argweight,
				 double maxiterations, int& solvedExactly, int factor,
                     int NUM_BOXES, Box* boxes) {
	Box outputBox = {-1, -1, -1, -1, -1.};
    argwidth += 1; // make space for 1 pixel padding
    argheight += 1;
   PyramidQualityFunction quality_bound;

// set up structure for pyramid grid parameters
// TODO: find a nicer way to handle the variable number of parameters
    const int numcells = argnumlevels*(argnumlevels+1)*(2*argnumlevels+1)/6;

    PyramidParameters paramstruct;
    paramstruct.numlevels=argnumlevels;
    paramstruct.weightptr = new double*[numcells];
    for (unsigned int i=0; i < numcells; i++) {
        paramstruct.weightptr[i] = &argweight[i*argnumclusters];
    }
    
// set up everything needed to calculate qualities and bounds
    quality_bound.setup(argnumpoints, argwidth, argheight, argxpos, argypos, argclst, &paramstruct);
    delete [] paramstruct.weightptr;

// intialize the search space (start with full image)
    sstate* single = new sstate(round_down(argwidth-1,factor), round_down(argheight-1,factor));
    sstate* best = new sstate(round_down(argwidth-1,factor), round_down(argheight-1,factor));

    sstate* single_original= single;
    sstate* best_original = best;
    best->upper =  -1e10;
//    sstate* fullspace = new sstate(round_down(argwidth-1,factor), round_down(argheight-1,factor));
    
// push first box set into priority queue
    //sstate_heap H;
    //H.push(fullspace);


		int upper_x =  round_down(argwidth-1,factor);
		int upper_y = round_down(argheight-1,factor);

		struct timeval start_time;
		gettimeofday(&start_time, NULL);
		struct timeval end_time;
		int counter = 0;

        for(int p = 0 ; p < NUM_BOXES ; p++)
        {
            single->only[0] = boxes[p].left;
            single->only[1] = boxes[p].top;
            single->only[2] = boxes[p].right;
            single->only[3] = boxes[p].bottom;
            single->upper = quality_bound.upper_bound(single);
            if(single->upper > best->upper)
            {
                sstate* temp = best;
                best = single;
                single = temp;
            }
            counter++;
         }

		if(best->upper<0)
		{
			best->upper = 0;
			for(int j = 0 ; j<4; j++)
				best->only[j] = 0;
		}
		gettimeofday(&end_time, NULL);
		double microseconds = 1e6 * (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec);
//		printf("We evaluated each guy in on average micro seconds %f %d\n", microseconds/counter, counter);
/*    while ((extract_split_and_insert(&H,quality_bound,factor) >= 0) && (counter < maxiterations)) {
        counter++;
				printf("Whats the count %d\n", counter);
    }*/

		if(counter<maxiterations)
			solvedExactly=1;
		else
			solvedExactly=0;
// at convergence or error, return result or best guess
    const sstate* curstate = best; //H.top();
    outputBox.left   = ((curstate->only[0]+curstate->only[0])>>1) -1;  // remove padding
    outputBox.top    = ((curstate->only[1]+curstate->only[1])>>1) -1;
    outputBox.right  = ((curstate->only[2]+curstate->only[2])>>1) -1;
    outputBox.bottom = ((curstate->only[3]+curstate->only[3])>>1) -1;
    outputBox.score  = curstate->upper;

// Free memory of the states that are still in the queue
/*        while (!H.empty()) {
        delete H.top();
            H.pop();
    }*/

// generic function to free any internal resource 
    quality_bound.cleanup();
		delete single_original;
		delete best_original;

    return outputBox;
}

}

#ifdef __MAIN__

static void usage(char *progname) {
    std::cerr << "usage: " << progname << " width height weight-file data-file\n";
    exit(1);
}

// helper routine to read ASCII data files 
// INPUT: filename: name of input file
// INPUT/OUTPUT: data: collection of vectors to be filled
static int readdata_NxM(const char* filename, std::vector< std::vector<double> > &data) {

    const unsigned int numcolumns=data.size();

// start with empty vectors
    for (unsigned int i=0;i<numcolumns;i++) {
        data[i].clear();
    }

    std::ifstream infile(filename);
    if (!infile)
        return -1;

    while (! infile.eof() ) {
        for (unsigned int i=0;i<numcolumns;i++) {
            double tmpval;
            infile >> tmpval;
            data[i].push_back(tmpval);
        }
    }
    infile.close();

// usually, eof occurs too late: we have read one extra entry
    for (unsigned int i=0;i<numcolumns;i++)
        data[i].pop_back();

// all vector should have same length
        const int numpts = data[0].size();
    return numpts;
}

// parse an int value from env variable
static int igetenv(const char* name, int defaultvalue, int low, int high) {
    int val = defaultvalue;
    if (getenv(name))
        return atoi(getenv(name));
    if (val < low)
        val=low;
    else if (val>high)
        val=high;
    return val;
}

// convenience function to control the behaviour through environment variables
static void set_parameters() {
    maxresults = igetenv("maxresults",1,1,10000);
    numlevels = igetenv("numlevels",1,1,100);
    verbose = igetenv("verbose",0,0,100000000);
    return;
}

int main(int argc, char* argv[]) {
    if (argc < 5)
        usage(argv[0]);

    set_parameters();
    sstate_heap H;

// first two arguments are width and height. 
    const int width = atoi(argv[1]);
    const int height = atoi(argv[2]);
    if ((width<2) || (width>MAXWIDTH) || (height<2) || (height>MAXHEIGHT))
       usage(argv[0]);

// read weight for clusters (1 column)
    std::vector< std::vector<double> > weightdata(1);
    int numweights = readdata_NxM(argv[3], weightdata);
    if (numweights <= 0) {
        std::cerr << "Error reading data from file \n" << argv[3] << std::endl;
        usage(argv[0]);
    }
    
    const int numcells = numlevels*(numlevels+1)*(2*numlevels+1)/6; // = 1+2^2+...+n^2
    const int numclusters = numweights/numcells;
    if (numclusters > MAXCLUSTERS) {
        std::cerr << "Can't handle that many clusters" << std::endl;
        usage(argv[0]);
    }


// read 3-column data file in format x,y,clusterID
    std::vector< std::vector<double> > rawdata(3);
    int datapts = readdata_NxM(argv[4], rawdata);
    if (datapts <= 0) {
        std::cerr << "Error reading data from file \n" << argv[4] << std::endl;
        usage(argv[0]);
    }

// convert from Nx3 matrix to individual vectors. Not very elegant...
    double* xpos = &rawdata[0][0];
    double* ypos = &rawdata[1][0];
    double* clst = &rawdata[2][0];
    double* weights = &weightdata[0][0];

// loop over target number of boxes
    for (int k=maxresults; k>0; --k) {
        Box bestBox = pyramid_search(datapts, width, height,  xpos, ypos, clst, 
                                     numclusters, numlevels, weights);
        std::cout << std::setprecision(12) << bestBox.score << " ";
        std::cout << bestBox.left << " ";
        std::cout << bestBox.top << " ";
        std::cout << bestBox.right << " ";
        std::cout << bestBox.bottom << " " ;

// before searching for next boxes, zero out the region of the box found.
// because removing elements from an array is cumbersome, we create a new one 
// from only points outside the box
        if (k>0) {
            std::vector< std::vector<double> > tmpdata(3);
            for (int i=0;i<datapts;i++) {
                if  ((xpos[i]<bestBox.left) || (xpos[i]>bestBox.right)
                    || (ypos[i]<bestBox.top) || (ypos[i]>bestBox.bottom)) {
                        tmpdata[0].push_back(xpos[i]);
                        tmpdata[1].push_back(ypos[i]);
                        tmpdata[2].push_back(clst[i]);
                }
            }
            datapts = tmpdata[0].size();
            for (int i=0;i<3;i++) {
                rawdata[i].resize(datapts);
                std::copy(tmpdata[i].begin(), tmpdata[i].end(), rawdata[i].begin());
            }
        }
    }
    std::cout << std::endl;
}
#endif
