CC = mpicc
CFLAGS = -Wall
LIBS = -lpthread -lm
TARGET = a2
NP = 13
ARGS = 3 4 5

all: $(TARGET)

$(TARGET): a2.c
	$(CC) $(CFLAGS) a2.c -o $(TARGET) $(LIBS)

run: $(TARGET)
	mpirun -np $(NP) -oversubscribe ./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)