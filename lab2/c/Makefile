
all: barrier group

%: %.c
	mpicc -o $@ $< -Wall -Wshadow -Wextra -std=c99 -lm

