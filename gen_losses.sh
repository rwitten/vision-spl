
cd output
FILES=`command ls *guesses* | grep -v .ap`
cd ..
for f in $FILES
do
    ./ap_compute.sh ${f} output/${f}.ap `pwd`
	echo "./ap_compute.sh ${f} output/${f}.ap"
done

cd output
FILES=`command ls *classify_output | grep -v .ap`
cd ..

for f in $FILES
do
	echo 'GETTING 0-1 LOSS FOR ' $f
	tail -n 1 output/$f  | awk '{print $6;}' > output/${f}.loss
done
