CXX = mpicxx
CXXFLAGS = -O2 -Wno-unused-result

TARGET = task
SRC = poisson_mpi_gpu_2.cu

OMPI_CXX = $(shell pwd)/hipcc_mpi
export OMPI_CXX

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f *.out
	rm -f *.err
	rm -f *.csv
	rm -f *.db
	rm -f *.json
	rm -f *.txt