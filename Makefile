CC=clang
FILES=src/*.c

WARNS=-std=c2x -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Werror=return-type
ifdef debug
CFLAGS=-lm -g3 $(sanitize:%=-fsanitize=%)
else
CFLAGS=-lm -DNDEBUG -O3 -march=native -flto -ffast-math -fno-reciprocal-math
endif

default:
	@$(CC) $(FILES) $(WARNS) $(CFLAGS) $(EXTRA)
