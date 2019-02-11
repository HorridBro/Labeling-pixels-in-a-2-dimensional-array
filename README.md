# Labeling-pixels-in-a-2-dimensional-array

## Compile
```
cmake --build cmake-build-debug --target main -- -j $(grep 'cpu cores' /proc/cpuinfo | uniq | awk -F ': ' '{print $2}')
```
## Run
```
 mpirun -np [proc_nr] cmake-build-debug/main image_path [-hsv] [-show_images]
```
## Prerequisites
### Opencv
 - https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh
### OpenMPI
- https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.3.tar.gz