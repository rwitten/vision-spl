./svm_bbox_learn -c 10 -o 0 --n 2 data/train_big.txt output/cccp10.model output/cccp10 > output/cccp10.log
./svm_bbox_learn -c 10 -o 0 -k 50 -m 1.3 --n 2 data/train_big.txt output/spl10.model output/spl10 > output/spl10.log
./svm_bbox_learn -z 1 -c 10 -o 0 -k 50 -m 1.3 --n 2 data/train_big.txt output/splplus10.model output/splplus10 > output/splplus10.log
