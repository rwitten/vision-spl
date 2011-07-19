#! /bin/csh -f

set basedir = "~rwitten/scratch/temp/spm/data/hog/train"

foreach line ( `cat /afs/cs.stanford.edu/u/rwitten/scratch/temp/spm/data/outputbbs.txt` )
  set num = `echo $line | gawk -F ":" '{print $1}'`
  set h = `echo $line | gawk -F ":" '{h = substr($2,2,index($2,",")-2); print h;}'`
		set w = `echo $line | gawk -F ":" '{w = substr($2,index($2,",")+1,length($2)-index($2,",")-1);print w}'`
		if ( ! -e $basedir/pos/${num}_spquantized_1000_hog.mat ) then
		    if ( ! -e $basedir:h/test/pos/${num}_spquantized_1000_hog.mat ) then
			echo "Error! $line"
			exit
		    endif
		    continue;
		endif
		
		set bboxfile = $basedir/pos/${num}_bbox.txt
		echo $bboxfile
		echo $h $w > $bboxfile
end
