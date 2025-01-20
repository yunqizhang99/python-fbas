# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g -MMD -MP -O3
LDFLAGS =

# Source files for each program
SRC1 = src/partitions.c
SRC2 = src/multi-partitions.c src/constellation.c

# Generate object and dependency files
OBJ1 = $(SRC1:.c=.o)
OBJ2 = $(SRC2:.c=.o)
DEP1 = $(OBJ1:.o=.d)
DEP2 = $(OBJ2:.o=.d)

# Target binaries
TARGET1 = partitions
TARGET2 = constellation

# Default target builds both programs
all: $(TARGET1) $(TARGET2)

# Rule to build program1
$(TARGET1): $(OBJ1)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to build program2
$(TARGET2): $(OBJ2)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Include dependency files
-include $(DEP1) $(DEP2)

# Clean up generated files
clean:
	rm -f $(OBJ1) $(OBJ2) $(DEP1) $(DEP2) $(TARGET1) $(TARGET2)

# Phony targets to avoid conflicts with files of the same name
.PHONY: all clean

