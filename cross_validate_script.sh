commands[1]=''
commands[2]='-m 1.3 -k 50 '
commands[3]='-m 1.3 -k 50 -z 1 '

names[1]='cccp'
names[2]='spl'
names[3]='splplus'

base_dir='.'
#base_dir='/afs/cs.stanford.edu/u/rwitten/projects/multi_kernel_spl'

for algorithm in 1  #this controls which algorithm we're doing
do
    for fold in 2
    do
        for randomness in 2
        do  
            command_train="./svm_bbox_learn -s $randomness -c 5000 -o 0 --n 2 ${commands[$algorithm]} $base_dir/data/train.${fold}.txt $base_dir/output/${names[$algorithm]}5000_${fold}_$randomness.model $base_dir/output/${names[$algorithm]}5000_${fold}_$randomness > $base_dir/output/${names[$algorithm]}5000_${fold}_$randomness.log"
            command_test_on_train="./svm_bbox_classify --n 2 $base_dir/data/train.${fold}.txt $base_dir/output/${names[$algorithm]}5000_${fold}_$randomness.model $base_dir/output/${names[$algorithm]}5000_${fold}_${randomness}.labels $base_dir/output/${names[$algorithm]}5000_${fold}_${randomness}.latent"
           
            command_test_on_test="./svm_bbox_classify --n 2 $base_dir/data/train.${fold}.txt $base_dir/output/${names[$algorithm]}5000_${fold}_$randomness.model $base_dir/output/${names[$algorithm]}5000_${fold}_$randomness.labels $base_dir/output/${names[$algorithm]}5000_${fold}_$randomness.latent"
#            matlab -r "cd /afs/cs.stanford.edu/u/rwitten/scratch/VOC2007/VOCdevkit; spl_compute_pr('bbox_cccp_c10000_unbalanced_j1.test.scores');exit;" | grep magic | awk '{print $2;}'
            echo $command_train
            echo $command_test_on_train
            echo $command_test_on_test
            #`$command_train`
        done
    done 
done
