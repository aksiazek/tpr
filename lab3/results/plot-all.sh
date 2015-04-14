for name in 'ekf' 'speedup' 'time'; do
        for p in 10000 50000 100000; do
                gnuplot -e "data='results/regular-$p'; out='results/$name-regular-$p.png'" results/plot-$name.gpl
                gnuplot -e "data='results/scaled-$p'; out='results/$name-scaled-$p.png'" results/plot-$name.gpl
        done
done
