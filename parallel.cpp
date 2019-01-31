#include "include/includes.h"
#include "include/utils.h"
#include "include/union_find.h"
#include "include/image.h"

#include <mpi.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

static int numprocs;

int main(int argc, char **argv) {
    int my_rank;

    MPI_Status status;
    MPI_Init (&argc, &argv);
    for(int i = 1; i < argc; ++i){
        string a = argv[i];
        cout << argv[i] << "\n";
    }
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    double time_start = MPI_Wtime();
    cout << "Hello World, my rank is " << my_rank <<" "<< MPI_Wtime() - time_start << std::endl;
    MPI_Finalize ();
    return 0;
}