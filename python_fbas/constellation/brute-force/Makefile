CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O3
LDFLAGS =

# List of source files
SRCS = constellation.c print-multi-partitions.c partitions.c
# List of object files
OBJS = $(SRCS:.c=.o)

all: constellation print-multi-partitions partitions

constellation: constellation.o multi-partitions.o
	$(CC) $(LDFLAGS) -o $@ $^

print-multi-partitions: print-multi-partitions.o multi-partitions.o
	$(CC) $(LDFLAGS) -o $@ $^

partitions: partitions.o
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean up build files
clean:
	rm -f $(OBJS) constellation print-multi-partitions partitions

.PHONY: all clean