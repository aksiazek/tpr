for points in 10000 50000 100000; do
        rm results/regular-$points results/scaled-$points
	ws_base=8
	for ws_factor in `seq 1 5`; do
		ws=$[ws_base * ws_factor]
		mpiexec -np $ws ./pi $points >> results/regular-$points
		mpiexec -np $ws ./pi $[points * ws_factor] >> results/scaled-$points
	done
done
