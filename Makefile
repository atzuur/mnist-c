CC=gcc
FILES=src/*.c

WARNS=-std=c23 -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Werror=return-type -Wno-unknown-pragmas
ifdef debug
CFLAGS=-lm -g3 $(sanitize:%=-fsanitize=%)
else
CFLAGS=-lm -DNDEBUG -O3 -march=native -flto -ffast-math
endif

default:
	@$(CC) $(FILES) $(WARNS) $(CFLAGS) $(EXTRA)
