#!/bin/bash

names[1]='cccp'
names[2]='spl'
names[3]='splplus'

base_dir='/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl'
cd $base_dir

for C in 100
do
  for classfold in 'car' #`cat classes`
	do
		for foldnum in 1 2 3
		do
			fold=${classfold}_${foldnum}
			fold_is_done=1
			for randomness in  1 2 3
			do
				for algorithm in   1 2 3  
				do
					basename=${names[$algorithm]}${C}_0_0_${fold}_$randomness
					if [ !  -f ./output/${basename}.endtime ]; then
						echo "missing ./output/${basename}.endtime"
						fold_is_done=0
					fi

					if [ -f ./output/${basename}.endtime ]; then
						echo "working " ./output/${basename}
						tail -n 1 ./output/${basename}.time | awk '{print $1;}' > ./output/${basename}.train_objective

						./ap_compute.sh output/${basename}.train_guesses output/${basename}.train_guesses.ap `pwd`
						./ap_compute.sh output/${basename}.test_guesses output/${basename}.test_guesses.ap `pwd`

						tail -n 1 output/${basename}.test_classify_output  | awk '{print $2;}' > output/${basename}.test_classify_output.loss
						tail -n 1 output/${basename}.train_classify_output  | awk '{print $2;}' > output/${basename}.train_classify_output.loss
						tail -n 1 output/${basename}.test_classify_output  | awk '{print $4;}' > output/${basename}.test_classify_output.weighted_loss
						tail -n 1 output/${basename}.train_classify_output  | awk '{print $4;}' > output/${basename}.train_classify_output.weighted_loss
					fi
				done
			done
		done
	done
done
