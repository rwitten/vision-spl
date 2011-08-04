#!/bin/bash

names[1]='cccp'
names[2]='spl'
names[3]='splplus'

base_dir='/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl'
cd $base_dir

for C in 100000
do
  for classfold in `cat classes`
	do
		fold=${classfold}_1
		fold_is_done=1
		for randomness in  1
		do
			for algorithm in   1 2 3  
			do
				if [ !  -f ./output/${names[$algorithm]}${C}_${fold}_$randomness.endtime ]; then
					echo "missing ./output/${names[$algorithm]}${C}_${fold}_$randomness.endtime"
					fold_is_done=0
				fi

				if [ -f ./output/${names[$algorithm]}${C}_${fold}_$randomness.endtime ]; then
					echo "working " ./output/${names[$algorithm]}${C}_${fold}_$randomness
					tail -n 1 ./output/${names[$algorithm]}${C}_${fold}_$randomness.time | awk '{print $1;}' > ./output/${names[$algorithm]}${C}_${fold}_$randomness.train_objective

					./ap_compute.sh output/${names[$algorithm]}${C}_${fold}_$randomness.train_guesses output/${names[$algorithm]}${C}_${fold}_$randomness.train_guesses.ap `pwd`
					./ap_compute.sh output/${names[$algorithm]}${C}_${fold}_$randomness.test_guesses output/${names[$algorithm]}${C}_${fold}_$randomness.test_guesses.ap `pwd`

					tail -n 1 output/${names[$algorithm]}${C}_${fold}_$randomness.test_classify_output  | awk '{print $2;}' > output/${names[$algorithm]}${C}_${fold}_$randomness.test_classify_output.loss
					tail -n 1 output/${names[$algorithm]}${C}_${fold}_$randomness.train_classify_output  | awk '{print $2;}' > output/${names[$algorithm]}${C}_${fold}_$randomness.train_classify_output.loss
					tail -n 1 output/${names[$algorithm]}${C}_${fold}_$randomness.test_classify_output  | awk '{print $4;}' > output/${names[$algorithm]}${C}_${fold}_$randomness.test_classify_output.weighted_loss
					tail -n 1 output/${names[$algorithm]}${C}_${fold}_$randomness.train_classify_output  | awk '{print $4;}' > output/${names[$algorithm]}${C}_${fold}_$randomness.train_classify_output.weighted_loss
				fi
			done
		done
	done
done
