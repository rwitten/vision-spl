#! /bin/csh -f

set basedir = "~rwitten/scratch/temp/spm/data/hog/train";
set outfile = "train_big.txt";

rm $outfile
foreach type ( pos neg )
  foreach file ( `ls $basedir/$type | grep mat` )
		  set num = `echo $file | gawk '{print substr($0,1,index($0,"_")-1)}'`
				set height = `head -n 2 $basedir/$type/$file | tail -n 1 | gawk '{print substr($0,2,index($0,",")-2)}'`
				set width = `head -n 2 $basedir/$type/$file | tail -n 1 | gawk '{print substr($0,index($0,",")+1,length($0)-index($0,",")-1)}'`
				if ( $type == "pos" ) then
								set typenum = 1;
				    set bboxfile = $basedir/$type/${num}_bbox.txt
				    set bbox = `cat $bboxfile`
				else
								set typenum = 0;
				    set bbox = "0 0"
				endif
				
				echo "train/$type/$num $typenum $height $width $bbox"
				echo "train/$type/$num $typenum $height $width $bbox" >> $outfile
		end
end
