#!/bin/bash

commands[1]=' --j 6 -l 1.5 '
commands[2]=' --j 6 -l 1.5 -m .1 -k 50 '
commands[3]=' --j 6 -l 1.5 -m .1 -k 50 -z 1 '

names[1]='cccp'
names[2]='spl'
names[3]='splplus'

base_dir='/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl'
cd $base_dir

./gen_losses.sh
for fold in 1 2 3 4 
do
	fold_is_done=1
    for randomness in  1 2 3 
    do
        for algorithm in  1 2 3 
        do
			if [ !  -f ./output/${names[$algorithm]}5000_${fold}_$randomness.endtime ]; then
				fold_is_done=0
		    fi
			if [ -f ./output/${names[$algorithm]}5000_${fold}_$randomness.endtime ]; then
				tail -n 1 ./output/${names[$algorithm]}5000_${fold}_$randomness.time | awk '{print $1;}' > ./output/${names[$algorithm]}5000_${fold}_$randomness.train_objective
			fi
        done
    done 

	echo $fold_is_done
	if [ $fold_is_done -eq 0 ]; then
		echo "FOLD  $fold ISN'T DONE"
	fi
	
	if [ $fold_is_done -gt 0 ]; then # this fold has completed.
		echo "FOLD $fold IS DONE"
		output_file="results_fold_"$fold
		echo '     objective       train_loss    test_loss     train_ap     test_ap' > $output_file
		
	    for algorithm in 1 2 3 
		do
			objective=` cat output/${names[$algorithm]}5000_${fold}_*.train_objective | awk '{if(min==""){min=$1};if($1<min){min=$1}}END{print min}'`
			train_loss=`cat output/${names[$algorithm]}5000_${fold}_*train*.loss | awk '{if(min==""){min=$1};if($1<min){min=$1}}END{print min}'`
			test_loss=`cat output/${names[$algorithm]}5000_${fold}_*test*.loss | awk '{if(min==""){min=$1};if($1<min){min=$1}}END{print min}'`
			train_ap=`cat output/${names[$algorithm]}5000_${fold}_*train*.ap | awk '{if(max==""){max=$1};if($1>max){max=$1}}END{print max}'`
            test_ap=`cat output/${names[$algorithm]}5000_${fold}_*test*.ap | awk '{if(max==""){max=$1};if($1>max){max=$1}}END{print max}'`
			echo ${names[$algorithm]} '    ' $objective '    ' $train_loss '    ' $test_loss '     ' $train_ap '     ' $test_ap >> $output_file
			echo printing
		done	
	fi
done
