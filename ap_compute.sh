cd $3
output_file=$2

matlab_dir=/afs/cs.stanford.edu/u/rwitten/scratch/VOC2007/VOCdevkit/

cp output/$1 ${matlab_dir}

matlab -nosplash -nojvm -r "cd $matlab_dir; spl_compute_pr('$1'); exit;" | grep magic | awk '{print $2;}' > $output_file

