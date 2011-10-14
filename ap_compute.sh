output_file=$2
input_file=$1
matlab_dir=/afs/cs.stanford.edu/u/rwitten/scratch/VOC2007/VOCdevkit/

matlab -nosplash -nojvm -r "cd $matlab_dir; spl_compute_pr('$input_file'); exit;" | grep magic | awk '{print $2;}' > $output_file
