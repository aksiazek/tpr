for name in 'ekf' 'speedup'; do
        for p in 100000 1000000 10000000; do
                gnuplot -e "data='regular-$p'; out='$name-regular-$p.png'" plot-$name.gpl
                gnuplot -e "data='scaled-$p'; out='$name-scaled-$p.png'" plot-$name.gpl
        done
done
