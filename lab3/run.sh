for points in 10000 50000 100000; do
        rm results/regular-$points results/scaled-$points
	ws_base=1
	for ws_factor in `seq 1 8`; do
		ws=$[ws_base * ws_factor]
		mpiexec -np $ws ./pi $points 0 >> results/regular-$points
		mpiexec -np $ws ./pi $points 1 >> results/scaled-$points
	done
done
