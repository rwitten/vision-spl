commands[1]=' --j 6 -l 1.5 '
commands[2]=' --j 6 -l 1.5 -m 1.3 -k 50 '
commands[3]=' --j 6 -l 1.5 -m 1.3 -k 50 -z 1 '

names[1]='cccp'
names[2]='spl'
names[3]='splplus'

base_dir='/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl'
cd $base_dir

for fold in 1
do
    for randomness in 1
    do
        for algorithm in 1 2 3
        do 
            command_starttimestamp="date > ./output/${names[$algorithm]}5000_${fold}_$randomness.starttime"
            command_endtimestamp="date > ./output/${names[$algorithm]}5000_${fold}_$randomness.endtime" 
            command_train="./svm_bbox_learn --s $randomness -c 5000 -o 0 --n 2 ${commands[$algorithm]} ./data/train.${fold}.txt ./output/${names[$algorithm]}5000_${fold}_$randomness.model ./output/${names[$algorithm]}5000_${fold}_$randomness > ./output/${names[$algorithm]}5000_${fold}_$randomness.train_output"
            command_test="./svm_bbox_classify --n 2 ./data/test.${fold}.txt ./output/${names[$algorithm]}5000_${fold}_$randomness.model ./output/${names[$algorithm]}5000_${fold}_${randomness}.labels ./output/${names[$algorithm]}5000_${fold}_${randomness}.latent  ./output/${names[$algorithm]}5000_${fold}_${randomness}.test_guesses >./output/${names[$algorithm]}5000_${fold}_${randomness}.test_classify_output"
#            command_test_matlab="./test_error.sh ${names[$algorithm]}5000_${fold}_${randomness}.test_guesses output/${names[$algorithm]}5000_${fold}_${randomness}.final_test_score $base_dir"

            command_test_on_train="./svm_bbox_classify --n 2 ./data/train.${fold}.txt ./output/${names[$algorithm]}5000_${fold}_$randomness.model ./output/${names[$algorithm]}5000_${fold}_${randomness}.labels_train ./output/${names[$algorithm]}5000_${fold}_${randomness}.latent_train ./output/${names[$algorithm]}5000_${fold}_${randomness}.train_guesses >./output/${names[$algorithm]}5000_${fold}_${randomness}.train_classify_output"
#            command_train_matlab="./test_error.sh ${names[$algorithm]}5000_${fold}_${randomness}.train_guesses output/${names[$algorithm]}5000_${fold}_${randomness}.final_train_score $base_dir"



            script_name="${names[$algorithm]}_${fold}_${randomness}.shell"
            echo '#!/bin/bash' > $script_name
            echo "cd $base_dir" >> $script_name
            echo $command_starttimestamp >> $script_name
            echo $command_train >> $script_name
            echo $command_test  >> $script_name
            echo $command_test_matlab >> $script_name
            echo $command_test_on_train  >> $script_name
            echo $command_train_matlab >> $script_name
            echo $command_endtimestamp >> $script_name
            chmod +x $script_name
            ~/bin/appendJob.pl ${base_dir}/${script_name}
        done
    done 
done
