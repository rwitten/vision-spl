names[1]='cccp'
names[2]='spl'
names[3]='splplus'

for fold in 1 2 
do
	for algorithm in 1 2 3
	do
		echo "Were doing " $fold for ${names[$algorithm]}
		./svm_bbox_classify --n 2 --j 4 ./data/test.${fold}.txt ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.model ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.labels_test ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.latent_test ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.test_guesses > ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.test_classify_output
		./svm_bbox_classify --n 2 --j 4 ./data/train.${fold}.txt ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.model ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.labels_train ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.latent_train ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.train_guesses > ./jul_13_overnight/${names[$algorithm]}5000_4_${fold}_1.train_classify_output
	done
done
