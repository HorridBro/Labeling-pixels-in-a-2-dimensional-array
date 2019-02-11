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
unordered_map<int, pair<int, int>> global_parent;


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


unordered_map<int, pair<int, int>> solve_tile(int* a, int rows, int cols){
    unordered_map<int, pair<int, int>> parent;
    for (int i = 0; i <  rows; ++i){
        for (int j = 0; j < cols; ++j){
            int idx = i * cols + j;
            if(!a[idx]){
                continue;
            }
            vector<int> neighbours;
            int m = numeric_limits<int>::max();
            for(auto d : directions){
                int ncol = j + d.second;
                int nrow = i + d.first;
                if ((ncol >= 0 && ncol < cols) && (nrow >= 0 && nrow < rows)){
                    int neighbour = a[nrow * cols + ncol];
                    if(neighbour){
                        neighbours.push_back(neighbour);
                        m = min(m, neighbour);
                    }
                }
            }
            if(neighbours.empty()){
                make_set(parent, a[idx]);
            }
            else {
                a[idx] = m;
                for (int n : neighbours){
                    union_sets(parent, m, find_root(parent, n));
                }
            }
        }
    }
    return parent;
}


void merge_tiles_arr(int* a, pair<int, int> start, pair<int, int> stop, int cols){
    for (int i = start.first ; i <= stop.first; ++i){
        for (int j = start.second; j <= stop.second; ++j){
            int idx = i * cols + j;
            if(!a[idx]){
                continue;
            }
            vector<int> neighbours;
            int m = numeric_limits<int>::max();
            for(auto d : all_directions){
                int ncol = j + d.second;
                int nrow = i + d.first;
                if ((ncol >= start.second && ncol <= stop.second) && (nrow >= start.first && nrow <= stop.first)){
                    int neighbour = a[nrow *cols + ncol];
                    if(neighbour){
                        neighbours.push_back(neighbour);
                        m = min(m, neighbour);
                    }
                }
            }
            if(neighbours.empty()){
                make_set(global_parent, a[idx]);
            }
            else {
                a[idx] = m;
                for (int n : neighbours){
                    union_sets(global_parent, m, find_root(global_parent, n));

                }
            }
        }
    }
}


void merge_tiles(int** a, pair<int, int> start, pair<int, int> stop){

    for (int i = start.first ; i <= stop.first; ++i){
        for (int j = start.second; j <= stop.second; ++j){
            if(!a[i][j]){
                continue;
            }
            vector<int> neighbours;
            int m = numeric_limits<int>::max();
            for(auto d : all_directions){
                int ncol = j + d.second;
                int nrow = i + d.first;
                if ((ncol >= start.second && ncol <= stop.second) && (nrow >= start.first && nrow <= stop.first)){
                    int neighbour = a[nrow][ncol];
                    if(neighbour){
                        neighbours.push_back(neighbour);
                        m = min(m, neighbour);
                    }
                }
            }
            if(neighbours.empty()){
                make_set(global_parent, a[i][j]);
            }
            else {
                a[i][j] = m;
                for (int n : neighbours){
                        if(global_parent.find(n) == global_parent.end()){
                            make_set(global_parent, n);
                        }
                        union_sets(global_parent, m, find_root(global_parent, n));
                }
            }
        }
    }
}


void merge_parent(int *parent, int sz){
    for(int i = 0 ; i < sz; i+= 3){
        global_parent[parent[i]] = {parent[i+1], parent[i+2]};
    }
}

void merge_parent_inv(unordered_map<int, pair<int, int>> & parent, int* par){
    int i = 1;
    for(auto x : parent){
        par[i] = x.first;
        par[i + 1] = x.second.first;
        par[i + 2] = x.second.second;
        i += 3;
    }
    par[0] = int(parent.size() * 3);
}


void connected_component_labeling_scatter(int* a, int rows, int cols){
    int rows_left = rows % numprocs;
    int size = rows * cols;
    int pieces = rows / numprocs;
    int piece_size = pieces * cols;
    int* initial_data = new int[numprocs * 2 + 3];
    int* sendcount = initial_data + 3;
    int* disps = initial_data + numprocs + 3;
    int* buff = new int[size];
    int* parent_serialized;
    initial_data[0] = pieces;
    initial_data[1] = cols;
    initial_data[2] = rows;
    fill(sendcount, sendcount + numprocs, piece_size);
    disps[0] = 0;
    sendcount[0] += rows_left * cols;
    int skip = sendcount[0];
    for(int i = 1; i < numprocs; i++){
        disps[i] = skip;
        skip += sendcount[i];
    }
    MPI_Bcast(initial_data, numprocs * 2 + 3, MPI_INT, 0, MPI_COMM_WORLD);
    parent_serialized = new int[size];
    MPI_Scatterv(a, sendcount, disps, MPI_INT, buff, sendcount[0], MPI_INT, 0, MPI_COMM_WORLD);
    global_parent = solve_tile(a, rows_left + pieces , cols);
    MPI_Gatherv(a, sendcount[0], MPI_INT, a, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(parent_serialized, sendcount[0], MPI_INT, parent_serialized, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);
    for(int i = piece_size + rows_left * cols ; i < size; i += piece_size){
         merge_parent(parent_serialized + i + 1, parent_serialized[i]);
    }
    for(int i = pieces + rows_left; i < rows ; i+= pieces){
        merge_tiles_arr(a, {i - 1, 0}, {i, cols - 1}, cols);
    }
    merge_parent_inv(global_parent, parent_serialized);
    MPI_Bcast(parent_serialized, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(a, sendcount, disps, MPI_INT, buff, sendcount[0], MPI_INT, 0, MPI_COMM_WORLD);
    int root_rows = rows_left + pieces;
    for (int i = 0 ; i < root_rows; ++i){
        for (int j = 0; j < cols; ++j){
            int idx = i * cols + j;
            if(!a[idx]){
                continue;
            }
                a[idx] = find_root(global_parent, a[idx]);
        }
    }

    MPI_Gatherv(a, sendcount[0], MPI_INT, a, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);
    delete [] initial_data;
    delete [] buff;
}
void connected_component_labeling_parallel(int* a, int rows, int cols) {
    connected_component_labeling_scatter(a, rows, cols);

}


int master_main(int argc, char* argv[]){
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
        string tile = "-tile_size=";
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
        run_type = "Parallel";
    }
    else {
        connected_component_labeling = connected_component_labeling_serial;
        run_type = "Serial";
    }
    string img_path = argv[1];
    Mat img = imread(img_path);
    Mat out = transform_image_2_binary(img, method);
    int rows = out.rows , cols = out.cols;
    int** data = convert_mat(out);
    int *data_array = new int[rows * cols];
    for(int i = 0 ; i < rows; i++){
        for(int j = 0 ; j < cols ; j++){
            data_array[i * cols + j] = data[i][j];
        }
    }
    float duration;
    if( run_type == "Parallel"){
        high_resolution_clock::time_point start = high_resolution_clock::now();
        connected_component_labeling_parallel(data_array, rows, cols);
        high_resolution_clock::time_point stop = high_resolution_clock::now();
        duration = duration_cast<chrono::duration<float>>( stop - start ).count();
        for(int i = 0 ; i < rows; i++){
            for(int j = 0 ; j < cols ; j++){
                data[i][j] = data_array[i * cols + j];
            }
        }
    } else{

        high_resolution_clock::time_point start = high_resolution_clock::now();
        connected_component_labeling(data, rows, cols);
        high_resolution_clock::time_point stop = high_resolution_clock::now();
        duration = duration_cast<chrono::duration<float>>( stop - start ).count();
    }
    cout << run_type <<  " with " << numprocs << " processes Connected Component Labeling Duration = " << duration << " seconds\n";
    string folder = string("results/") + run_type + "/" + img_path.substr(img_path.find("/") + 1, img_path.find(".") - img_path.find("/") -1); ;
    uchar* colored = color_labels(data, rows, cols);
    Mat final = create_output_mat(colored, rows, cols);
    imwrite("Intermediary" + run_type + ".jpg", out);
    imwrite("Output"  + run_type + ".jpg", final);
    imwrite(folder + "Intermediary.jpg", out);
    imwrite(folder +"Output.jpg", final);

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
    delete [] data_array;
    delete_matrix(data, rows);
    return 0;
}


int slave_main(int rank){
    int *initial_data, rows, cols, size, i = 1, parent_size, *parent_serialized, *a, *b, *par_buff, *sendcount, *disps;
    initial_data = new int [numprocs * 2 + 3];
    MPI_Bcast(initial_data, numprocs * 2 + 3, MPI_INT, 0, MPI_COMM_WORLD);
    rows = initial_data[0];
    cols = initial_data[1];
    int total_rows = initial_data[2];
    sendcount = initial_data + 3;
    disps = initial_data + numprocs + 3;
    size = rows * cols;
    a = new int[size];
    b = new int[size];
    MPI_Scatterv(b, sendcount, disps, MPI_INT, a, size, MPI_INT, 0, MPI_COMM_WORLD);
    unordered_map<int, pair<int, int>> parent = solve_tile(a, rows, cols);
    int par_sz = int(parent.size() * 3);
    parent_size = max(par_sz, size);
    parent_serialized = new int[parent_size];
    parent_serialized[0] = par_sz;
    for(auto x : parent){
        parent_serialized[i] = x.first;
        parent_serialized[i + 1] = x.second.first;
        parent_serialized[i + 2] = x.second.second;
        i += 3;
    }
    MPI_Gatherv(a, size, MPI_INT, b, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(parent_serialized, size, MPI_INT, b, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);
    int total_size = total_rows * cols;
    int* all_parent = new int[total_size];
    MPI_Bcast(all_parent, total_size, MPI_INT, 0, MPI_COMM_WORLD);
    merge_parent(all_parent + 1, all_parent[0]);
    MPI_Scatterv(b, sendcount, disps, MPI_INT, a, size, MPI_INT, 0, MPI_COMM_WORLD);
    for (int k = 0 ; k < rows; ++k){
        for (int j = 0; j < cols; ++j){
            int idx = k * cols + j;
            if(!a[idx]){
                continue;
            }
                a[idx] = find_root(global_parent, a[idx]);
        }
    }
    MPI_Gatherv(a, size, MPI_INT, b, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);
    delete [] a;
    delete [] b;
    delete [] initial_data;
    delete [] parent_serialized;
    return 0;
}


int main(int argc, char* argv[]) {
    int my_rank = 0, ret_code = 0;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    if (my_rank == 0){
        ret_code = master_main(argc, argv);
    } else{
        ret_code = slave_main(my_rank);
    }
    MPI_Finalize ();
    return ret_code;
}