//mpirun  -np 1 cmake-build-debug/main images/gray/aloi.jpg -show_images //serial
//mpirun  -np 8 cmake-build-debug/main images/gray/aloi.jpg -show_images //parallel

#include "include/includes.h"
#include "include/utils.h"
#include "include/union_find.h"
#include "include/image.h"
#include <mpi.h>


using namespace std;
using namespace cv;
using namespace std::chrono;

static int numprocs;


void connected_component_labeling_serial(int** a, int rows, int cols){
    unordered_map<int, pair<int, int>> parent;
    for (int i = 0; i <  rows; ++i){
        for (int j = 0; j < cols; ++j){
            if(!a[i][j]){
                continue;
            }
            vector<int> neighbours;
            int m = numeric_limits<int>::max();
            for(auto d : directions){
                int ncol = j + d.second;
                int nrow = i + d.first;
                if ((ncol >= 0 && ncol < cols) && (nrow >= 0 && nrow < rows)){
                    int neighbour = a[nrow][ncol];
                    if(neighbour){
                        neighbours.push_back(neighbour);
                        m = min(m, neighbour);
                    }
                }
            }
            if(neighbours.empty()){
                make_set(parent, a[i][j]);
            }
            else {
                a[i][j] = m;
                for (int n : neighbours){
                    union_sets(parent, m, find_root(parent, n));
                }
            }
        }
    }
    for (int i = 0; i <  rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if(!a[i][j]){
                continue;
            }
            a[i][j] = find_root(parent, a[i][j]);
        }
    }
}


void connected_component_labeling_parallel(int** a, int rows, int cols){



}


int master_main(int argc, char* argv[]){
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    int show_images = 0,  method = 0;
    string run_type;
    void (*connected_component_labeling)(int**, int, int);
    string usage = "Usage:\t" + string(argv[0]) +" image_path [-hsv] [-show_images]\n";
    if (argc < 2){
        cout << "Error, invalid args! " << usage;
        return 1;
    }
    for(int i = 2; i < argc; ++i){
        string ar = argv[i];
        if (ar == "-hsv"){
            method = 1;
        }
        else if( ar == "-show_images"){
            show_images = 1;
        }
        else if(ar == "-h"){
            cout << usage;
            return 0;
        }
    }
    if(numprocs > 1){
        connected_component_labeling = connected_component_labeling_parallel;
        run_type = "Parallel";
    }
    else {
        connected_component_labeling = connected_component_labeling_serial;
        run_type = "Serial";
    }
    Mat img = imread(argv[1]);
    Mat out = transform_image_2_binary(img, method);
    int rows = out.rows , cols = out.cols;
    int** data = convert_mat(out);
    high_resolution_clock::time_point start = high_resolution_clock::now();
    connected_component_labeling(data, rows, cols);
    high_resolution_clock::time_point stop = high_resolution_clock::now();
    float duration = duration_cast<chrono::duration<float>>( stop - start ).count();
    cout << run_type << " Connected Component Labeling Duration = " << duration << " seconds\n";
    uchar* colored = color_labels(data, rows, cols);
    Mat final = create_output_mat(colored, rows, cols);
    imwrite("./black-white.jpg", out);
    imwrite("./output.jpg", final);
    if (show_images){
        imshow("Input", img);
        imshow("Intermediary", out);
        imshow("Output", final);
        while(waitKey(1) != 27); //Esc key
        img.release();
        out.release();
        final.release();
    }
    delete [] colored;
    delete_matrix(data, rows);
    return 0;
}


int slave_main(int rank){



    return 0;
}


int main(int argc, char* argv[]) {
    int my_rank = 0, ret_code = 0;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0){
        ret_code = master_main(argc, argv);
    } else{
        ret_code = slave_main(my_rank);
    }

    MPI_Finalize ();
    return ret_code;
}