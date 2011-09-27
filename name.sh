# invocation is ./name.sh ${algorithm} ${class} ${C} ${foldnum} ${randomness} ${hallu} ${kernel} ${prox_weight} ${epsilon}

algorithm=$1
class=$2
C=$3
foldnum=$4
randomness=$5
hallu=$6
kernel=$7
prox_weight=$8
epsilon=$9

algorithms[1]='cccp'
algorithms[2]='spl'
algorithms[3]='splplus'

hallus[0]='nohallu'
hallus[1]='hallu'

kernels[0]='bow'
kernels[1]='spm'

echo ${algorithms[$algorithm]}_${class}_C${C}_fold${foldnum}_rand${randomness}_${hallus[$hallu]}_${kernels[$kernel]}_lambda${prox_weight}_eps${epsilon}

