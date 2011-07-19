#!/bin/bash

commands[1]=' --j 6 -l 1.5 '
commands[2]=' --j 6 -l 1.5 -m .1 -k 50 '
commands[3]=' --j 6 -l 1.5 -m .1 -k 50 -z 1 '

names[1]='cccp'
names[2]='spl'
names[3]='splplus'

base_dir='/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl'
cd $base_dir

for C in 100000
do
	for fold in   1 2 3 4 5
	do
		fold_is_done=1
		for randomness in  2 3 4
		do
			for algorithm in   1 2 3  
			do
				if [ !  -f ./output/${names[$algorithm]}${C}_6_${fold}_$randomness.endtime ]; then
					fold_is_done=0
				fi

				if [ -f ./output/${names[$algorithm]}${C}_6_${fold}_$randomness.endtime ]; then
					echo "working " ./output/${names[$algorithm]}${C}_6_${fold}_$randomness
					tail -n 1 ./output/${names[$algorithm]}${C}_6_${fold}_$randomness.time | awk '{print $1;}' > ./output/${names[$algorithm]}${C}_6_${fold}_$randomness.train_objective

					./ap_compute.sh output/${names[$algorithm]}${C}_6_${fold}_$randomness.train_guesses output/${names[$algorithm]}${C}_6_${fold}_$randomness.train_guesses.ap `pwd`
					./ap_compute.sh output/${names[$algorithm]}${C}_6_${fold}_$randomness.test_guesses output/${names[$algorithm]}${C}_6_${fold}_$randomness.test_guesses.ap `pwd`

					tail -n 1 output/${names[$algorithm]}${C}_6_${fold}_$randomness.test_classify_output  | awk '{print $2;}' > output/${names[$algorithm]}${C}_6_${fold}_$randomness.test_classify_output.loss
					tail -n 1 output/${names[$algorithm]}${C}_6_${fold}_$randomness.train_classify_output  | awk '{print $2;}' > output/${names[$algorithm]}${C}_6_${fold}_$randomness.train_classify_output.loss
					tail -n 1 output/${names[$algorithm]}${C}_6_${fold}_$randomness.test_classify_output  | awk '{print $4;}' > output/${names[$algorithm]}${C}_6_${fold}_$randomness.test_classify_output.weighted_loss
					tail -n 1 output/${names[$algorithm]}${C}_6_${fold}_$randomness.train_classify_output  | awk '{print $4;}' > output/${names[$algorithm]}${C}_6_${fold}_$randomness.train_classify_output.weighted_loss

				fi
			done
		done 

		if [ $fold_is_done -eq 0 ]; then
			echo "FOLD  $fold  and C=$C ISN'T DONE"
		fi
		
		if [ $fold_is_done -gt 0 ]; then # this fold has completed.
			echo "FOLD $fold and C=$C IS DONE"
			output_file="results_fold_"${C}_$fold
			echo '         objective       train_loss    train_weighted_loss    test_loss    test_weighted_loss    train_ap     test_ap' > $output_file
			
			for algorithm in 1 2 3 
			do
				objective=` cat output/${names[$algorithm]}${C}_6_${fold}_*.train_objective | awk '{if(min==""){min=$1};if($1<min){min=$1}}END{print min}'`
				train_loss=`cat output/${names[$algorithm]}${C}_6_${fold}_*train*.loss | awk '{if(min==""){min=$1};if($1<min){min=$1}}END{print min}'`
				test_loss=`cat output/${names[$algorithm]}${C}_6_${fold}_*test*.loss | awk '{if(min==""){min=$1};if($1<min){min=$1}}END{print min}'`
				train_weighted_loss=`cat output/${names[$algorithm]}${C}_6_${fold}_*train*.weighted_loss | awk '{if(min==""){min=$1};if($1<min){min=$1}}END{print min}'`
				test_weighted_loss=`cat output/${names[$algorithm]}${C}_6_${fold}_*test*.weighted_loss | awk '{if(min==""){min=$1};if($1<min){min=$1}}END{print min}'`
				train_ap=`cat output/${names[$algorithm]}${C}_6_${fold}_*train*.ap | awk '{if(max==""){max=$1};if($1>max){max=$1}}END{print max}'`
				test_ap=`cat output/${names[$algorithm]}${C}_6_${fold}_*test*.ap | awk '{if(max==""){max=$1};if($1>max){max=$1}}END{print max}'`
				echo ${names[$algorithm]} '    ' $objective '    ' $train_loss '    ' $train_weighted_loss '                ' $test_loss '    ' $test_weighted_loss '          ' $train_ap '     ' $test_ap >> $output_file
			done	
		fi
	done
done
