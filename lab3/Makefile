all: pi

%: %.c
	mpicc -o $@ $< -Wall -Wshadow -Wextra -std=c99 -lgmp

