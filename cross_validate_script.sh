j=6
C=2500

commands[1]=" --j $j -l 1.5 "
commands[2]=" --j $j -l 1.5 -m .1 -k 50 "
commands[3]=" --j $j -l 1.5 -m .1 -k 50 -z 1 "

commands_test[1]=" --j $j "
commands_test[2]=" --j $j "
commands_test[3]=" --j $j "

names[1]='cccp'
names[2]='spl'
names[3]='splplus'

base_dir=`pwd`
cd $base_dir

for randomness in 2
do
	for fold in 3 
	do
		for C in 100000 
		do
			for algorithm in   1 2 3 
			do
				if [  -f ./output/${names[$algorithm]}${C}_${fold}_$randomness.model ]; then
					echo "Skipping " ./output/${names[$algorithm]}${C}_${fold}_$randomness.model
					continue
				fi
				command_starttimestamp="date > ./output/${names[$algorithm]}${C}_${j}_${fold}_$randomness.starttime"
				command_endtimestamp="date > ./output/${names[$algorithm]}${C}_${j}_${fold}_$randomness.endtime" 
				command_train="./svm_bbox_learn --s $randomness -c ${C} -o 0 --n 2 ${commands[$algorithm]} ./data/train.${fold}.txt ./output/${names[$algorithm]}${C}_${j}_${fold}_$randomness.model ./output/${names[$algorithm]}${C}_${j}_${fold}_$randomness > ./output/${names[$algorithm]}${C}_${j}_${fold}_$randomness.train_output"
				command_test="./svm_bbox_classify --n 2 ${commands_test[$algorithm]} ./data/test.${fold}.txt ./output/${names[$algorithm]}${C}_${j}_${fold}_$randomness.model ./output/${names[$algorithm]}${C}_${j}_${fold}_${randomness}.labels ./output/${names[$algorithm]}${C}_${j}_${fold}_${randomness}.latent.test  ./output/${names[$algorithm]}${C}_${j}_${fold}_${randomness}.test_guesses >./output/${names[$algorithm]}${C}_${j}_${fold}_${randomness}.test_classify_output"

				command_test_on_train="./svm_bbox_classify --n 2 ${commands_test[$algorithm]} ./data/train.${fold}.txt ./output/${names[$algorithm]}${C}_${j}_${fold}_$randomness.model ./output/${names[$algorithm]}${C}_${j}_${fold}_${randomness}.labels_train ./output/${names[$algorithm]}${C}_${j}_${fold}_${randomness}.latent.train ./output/${names[$algorithm]}${C}_${j}_${fold}_${randomness}.train_guesses >./output/${names[$algorithm]}${C}_${j}_${fold}_${randomness}.train_classify_output"

				script_name="${names[$algorithm]}${C}_${j}_${fold}_${randomness}.shell"
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
#				~/bin/appendJob.pl ${base_dir}/${script_name}
			done
		done 
	done
done
