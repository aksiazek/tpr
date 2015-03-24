for name in 'results-own-bcast' 'results-bcast' 'results-own-reduce' 'results-reduce'; do
    gnuplot -e "data='c-results/$name.dat'; out='c-results/$name.png'" plot.gpl
done
