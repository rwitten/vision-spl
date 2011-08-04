#! /bin/csh -f

set basedir = "~rwitten/scratch/temp/spm/data/hog/test";
set outfile = "test_big.txt";

rm $outfile
foreach type ( pos neg )
  if ( $type == "neg" ) then
    foreach num (`cat test_negatives.txt`)
	set file = `ls $basedir/$type | grep $num`;
	set height = `head -n 2 $basedir/$type/$file | tail -n 1 | gawk '{print substr($0,2,index($0,",")-2)}'`
	set width = `head -n 2 $basedir/$type/$file | tail -n 1 | gawk '{print substr($0,index($0,",")+1,length($0)-index($0,",")-1)}'`
	echo "test/$type/$num 0 $height $width 0 0" >> $outfile
	end
    else
    foreach line (`cat /afs/cs.stanford.edu/u/rwitten/scratch/VOC2007/VOCdevkit/test_easy_postives.txt`)
	set num = `echo $line | awk '{split($0,a,":"); print a[1]}'`;
	set phrase = `echo $line | awk '{split($0,a,":"); print a[2]}'`;
	set bbox_height = `echo $phrase | gawk '{print substr($0,2,index($0,",")-2)}'`;
	set bbox_width = `echo $phrase | gawk '{print substr($0,index($0,",")+1,length($0)-index($0,",")-1)}'`;
        set file = `ls $basedir/$type | grep $num`;
	set height = `head -n 2 $basedir/$type/$file | tail -n 1 | gawk '{print substr($0,2,index($0,",")-2)}'`
        set width = `head -n 2 $basedir/$type/$file | tail -n 1 | gawk '{print substr($0,index($0,",")+1,length($0)-index($0,",")-1)}'`
	echo "test/$type/$num 1 $height $width $bbox_height $bbox_width" >> $outfile
	end
    endif
end
