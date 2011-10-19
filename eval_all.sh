for file in output/*model
do
  if [ -f $file ] ; then
    # name without extension
    basename=${file%\.*}
    echo $basename
    if [ !  -f ./${basename}.endtime ]; then
        echo "missing ./output/${basename}.endtime"
        fold_is_done=0
    fi

    if [ -f ./${basename}.endtime ]; then
        if [ ! -f ./${basename}.train_objective ]; then
            tail -n 1 ./${basename}.time | awk '{print $1;}' > ./${basename}.train_objective
            
            full_path=`pwd`
            echo looking for $full_path/${basename}.test_guesses
            ./ap_compute.sh $full_path/${basename}.train_guesses $full_path/${basename}.train_guesses.ap
            ./ap_compute.sh $full_path/${basename}.test_guesses $full_path/${basename}.test_guesses.ap

            tail -n 1 ${basename}.test_classify_output  | awk '{print $2;}' > ${basename}.test_classify_output.loss
            tail -n 1 ${basename}.train_classify_output  | awk '{print $2;}' > ${basename}.train_classify_output.loss
            tail -n 1 ${basename}.test_classify_output  | awk '{print $4;}' > ${basename}.test_classify_output.weighted_loss
            tail -n 1 ${basename}.train_classify_output  | awk '{print $4;}' > ${basename}.train_classify_output.weighted_loss
        fi 
    fi
  fi ;
done
