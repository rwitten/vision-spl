
cd output
FILES=`command ls *guesses* | grep -v .ap`
cd ..
for f in $FILES
do
    ./test_error.sh ${f} output/${f}.ap `pwd`
	echo "./test_error.sh ${f} output/${f}.ap"
done
