CC=gcc
CFLAGS=-Wextra -Wall -Werror -Wvla -pedantic -std=c99 -fsanitize=address -g
# -fsanitize=address allows us to get error messages regarding memory leaks
#  and buffer overflow.
#
# -g allows us to debug using gdb

SRC=main.c lstm.c

all:
	$(CC) $(CFLAGS) -o main $(SRC)

clean: all
	rm main main.o lstm.o
