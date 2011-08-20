names[1]='cccp'
names[2]='spl'
names[3]='splplus'

base_dir=`pwd`
cd $base_dir

for randomness in 1 
do
	for classfold in 'small'
	do
		for l in 0 1
		do
			num_neg=`cat data/train.${classfold}_1.txt | grep ' 0 ' | wc -l`
			num_pos=`cat data/train.${classfold}_1.txt | grep ' 1 ' | wc -l`
			j=`echo $num_neg $num_pos | awk '{print $1/$2;}'`	
			commands[1]=" --j $j --l $l "
			commands[2]=" --j $j --l $l -m .1 -k 50 "
			commands[3]=" --j $j --l $l -m .1 -k 50 -z 1 "
			commands_test[1]=" --j $j --l $l "
			commands_test[2]=" --j $j --l $l "
			commands_test[3]=" --j $j --l $l "
			
			for C in 500 1000 5000 10000
			do
				for algorithm in  1 
				do
					for foldnum in 1
					do
						fold=${classfold}_${foldnum}
						if [  -f ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness.starttime ]; then
							echo "Skipping " ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness
							continue
						fi
						command_starttimestamp="date > ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness.starttime"
						command_endtimestamp="date > ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness.endtime" 
						command_train="./svm_bbox_learn --s $randomness -c ${C} -o 0 --n 2 ${commands[$algorithm]} ./data/train.${fold}.txt ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness.model ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness > ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness.train_output"
						command_test="./svm_bbox_classify --c $C --n 2 ${commands_test[$algorithm]} ./data/test.${fold}.txt ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness.model ./output/${names[$algorithm]}${C}_${l}_${fold}_${randomness}.labels ./output/${names[$algorithm]}${C}_${l}_${fold}_${randomness}.latent.test  ./output/${names[$algorithm]}${C}_${l}_${fold}_${randomness}.test_guesses >./output/${names[$algorithm]}${C}_${l}_${fold}_${randomness}.test_classify_output"

						command_test_on_train="./svm_bbox_classify --c $C --n 2 ${commands_test[$algorithm]} ./data/train.${fold}.txt ./output/${names[$algorithm]}${C}_${l}_${fold}_$randomness.model ./output/${names[$algorithm]}${C}_${l}_${fold}_${randomness}.labels_train ./output/${names[$algorithm]}${C}_${l}_${fold}_${randomness}.latent.train ./output/${names[$algorithm]}${C}_${l}_${fold}_${randomness}.train_guesses >./output/${names[$algorithm]}${C}_${l}_${fold}_${randomness}.train_classify_output"

						script_name="${names[$algorithm]}${C}_${l}_${fold}_${randomness}.shell"
						echo '#!/bin/bash' > $script_name
						echo 'source ~/.bashrc' > $script_name
						echo "cd $base_dir" >> $script_name
						echo $command_starttimestamp >> $script_name
						echo $command_train >> $script_name
						echo $command_test  >> $script_name
						echo $command_test_matlab >> $script_name
						echo $command_test_on_train  >> $script_name
						echo $command_train_matlab >> $script_name
						echo $command_endtimestamp >> $script_name
						chmod +x $script_name
						echo "Posting job " ${base_dir}/${script_name}
						~/bin/appendJob.pl ${base_dir}/${script_name}
					done
				done
			done
		done 
	done
done
