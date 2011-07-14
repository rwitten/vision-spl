rm output/*
rm *.shell
rm ~/jobsCompleted.txt
touch ~/jobsCompleted.txt
rm ~/jobsQueued.txt
touch ~/jobsQueued.txt
rm ~/jobsRunning.txt
touch ~/jobsRunning.txt

./cross_validate_script.sh
