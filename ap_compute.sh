cd $3
output_file=$2

matlab_dir=/afs/cs.stanford.edu/u/rwitten/scratch/VOC2007/VOCdevkit/

cp $1 ${matlab_dir}

base_name=`basename $1`
matlab -nosplash -nojvm -r "cd $matlab_dir; spl_compute_pr('$base_name'); exit;" | grep magic | awk '{print $2;}' > $output_file
cd $matlab_dir
rm $base_name

