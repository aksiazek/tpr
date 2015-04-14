for points in 1000000 5000000 10000000; do
        rm regular-$points scaled-$points
	ws_base=12
	for ws_factor in `seq 1 5`; do
		ws=$[ws_base * ws_factor]
		mpiexec -np $ws ./pi $points >> regular-$points
		mpiexec -np $ws ./pi $[points * ws_factor] >> scaled-$points
	done
done
