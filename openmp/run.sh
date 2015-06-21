for points in 100000 1000000 10000000; do
        rm results/regular-$points results/scaled-$points
	./bucket $points 12 0 > results/regular-$points
	./bucket $points 12 1 > results/scaled-$points
done
