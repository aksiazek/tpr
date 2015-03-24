size=$1
name="c-results/group-$size"
echo $name
gnuplot -e "data='$name.dat'; out='$name.png'" plot.gpl

