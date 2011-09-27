base_dir=`pwd`
cd $base_dir

for randomness in 1
do
	for classfold in  'car'
	do
		for foldnum in 1
		do
			for C in 100 1000 5000 10000 50000 75000 100000 125000 250000 500000 1000000
			do
				for l in 0
				do
					for h in 0 1
					do
						for algorithm in  1
						do
							for prox_weight in 10
                            do
                                for epsilon in .01 .001
                                do
                                    num_neg=`cat data/train.${classfold}_1.txt | grep ' 0 ' | wc -l`
                                    num_pos=`cat data/train.${classfold}_1.txt | grep ' 1 ' | wc -l`

                                    basename=`./name.sh ${algorithm} ${classfold} ${C} ${foldnum} ${randomness} ${h} ${l} ${prox_weight} ${epsilon}`
                                    j=`echo $num_neg $num_pos | awk '{print $1/$2;}'`	
                                    commands[1]=" -e ${epsilon} --h $h --j $j --l $l --p ${prox_weight} "
                                    commands[2]=" -e ${epsilon} --h $h --j $j --l $l --p ${prox_weight} -m .1 -k 50 "
                                    commands[3]=" -e ${epsilon} --h $h --j $j --l $l --p ${prox_weight} -m .1 -k 50 -z 1 "
                                    commands_test[1]=" --j $j --l $l "
                                    commands_test[2]=" --j $j --l $l "
                                    commands_test[3]=" --j $j --l $l "
                                    fold=${classfold}_${foldnum}
                                    command_starttimestamp="date > ./output/${basename}.starttime"
                                    command_endtimestamp="date > ./output/${basename}.endtime" 
                                    command_train="./svm_bbox_learn --s $randomness -c ${C} -o 0 --n 2 ${commands[$algorithm]} ./data/train.${fold}.txt ./output/${basename}.model ./output/${basename} > ./output/${basename}.train_output"
                                    command_test="./svm_bbox_classify --c $C --n 2 ${commands_test[$algorithm]} ./data/test.${fold}.txt ./output/${basename}.model ./output/${basename}.labels ./output/${basename}.latent.test  ./output/${basename}.test_guesses >./output/${basename}.test_classify_output"

                                    command_test_on_train="./svm_bbox_classify --c $C --n 2 ${commands_test[$algorithm]} ./data/train.${fold}.txt ./output/${basename}.model ./output/${basename}.labels_train ./output/${basename}.latent.train ./output/${basename}.train_guesses >./output/${basename}.train_classify_output"
                                    
                                    command_starttime='START=$(date +%s)'
                                    command_endtime='END=$(date +%s)'
                                    command_difference='DIFF=$(( $END - $START ))'
                                    command_time_passed="echo \${DIFF} > ./output/${basename}.totaltime"
                                    script_name="${basename}.shell"
                                    echo '#!/bin/bash' > $script_name
                                    echo $command_starttime >> $script_name
                                    echo 'source ~/.bashrc' >> $script_name
                                    echo "cd $base_dir" >> $script_name
                                    echo $command_starttimestamp >> $script_name
                                    echo $command_train >> $script_name
                                    echo $command_test  >> $script_name
                                    echo $command_test_matlab >> $script_name
                                    echo $command_test_on_train  >> $script_name
                                    echo $command_train_matlab >> $script_name
                                    echo $command_endtimestamp >> $script_name
                                    echo $command_endtime >> $script_name
                                    echo $command_difference >> $script_name
                                    echo $command_time_passed >> $script_name

                                    chmod +x $script_name
                                    if [  -f ./output/${basename}.starttime ]; then
                                        echo "Skipping " ./output/${basename}
                                        continue
                                    fi
                                    echo "Posting job " ${base_dir}/${script_name}
                                    ~/bin/appendJob.pl ${base_dir}/${script_name}
                                done
                            done
						done
					done
				done
			done
		done 
	done
done
