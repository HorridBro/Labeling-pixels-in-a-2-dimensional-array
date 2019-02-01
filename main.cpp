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
static int dest = 1;
static int TILE_SIZE = 64;
const int LABEL_TAG = 1;
const int END_TAG = 2;
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


void connected_component_labeling_parallel_util(int** a, int rows, int cols, pair<int, int> start, pair<int, int> stop){
    int size = rows * cols;
    if(size <= TILE_SIZE){
        int* msg_to_send = new int[size + 3];
        int *temp_array = msg_to_send + 3;
        msg_to_send[0] = rows;
        msg_to_send[1] = cols;
        msg_to_send[2] = start.first;
        msg_to_send[3] = start.second;
        for(int i = start.first, ii = 0; i <= stop.first; i++, ii++){
            for(int j = start.first, jj = 0; j <= stop.second; j++, jj++){
                temp_array[ii * rows + jj] = a[i][j];
            }
        }
        MPI_Send(msg_to_send, size + 2, MPI_INT, dest, LABEL_TAG, MPI_COMM_WORLD);
        dest = (dest + 1) % numprocs + 1;
    }
    else{
        int new_rows = rows / 4;
        int new_cols = cols / 4;
        int rows_left = new_rows;
        int cols_left = new_cols;
        int mod_rows = rows % 4, mod_cols = cols % 4;
        if (mod_rows){
            rows_left += rows % 4;
        }
        if (mod_cols){
            cols_left += cols % 4;
        }
        connected_component_labeling_parallel_util(a, new_rows + 1, new_cols + 1, start , {start.first + new_rows, start.second + new_cols});
        connected_component_labeling_parallel_util(a, new_rows + 1, cols_left + 1, {start.first, start.second + new_cols + 1}, {start.first + new_rows, start.second + new_cols + cols_left + 1});
        connected_component_labeling_parallel_util(a, rows_left + 1, new_cols + 1, {start.first + new_rows + 1, start.second}, {start.first + new_rows + rows_left, start.second + new_cols});
        connected_component_labeling_parallel_util(a, rows_left + 1, cols_left + 1, {start.first + new_rows + 1, start.second + new_cols + 1} , stop);
        if(size / 4 <= TILE_SIZE){
            int recv_size = 3 + TILE_SIZE * 4;
            int* tiles = new int[recv_size *4];
            for(int i = 0 ; i < 4; ++i){
                int tile [recv_size];
                MPI_Recv(tile, recv_size, MPI_INT, MPI_ANY_SOURCE, LABEL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int * tl = tile + 3;
                if(tile[0] == start.first){
                    if(tile[1] == start.second){
                        copy(tl, tl+ recv_size, tiles);
                    } else{
                        copy(tl, tl + recv_size, tiles + recv_size);
                    }
                } else{
                    if(tile[1] == start.second){
                        copy(tl, tl + recv_size, tiles + 2 * recv_size);
                    } else{
                        copy(tl, tl + recv_size, tiles + 3 * recv_size);
                    }
                }
            }
        }
    }


}


void connected_component_labeling_parallel(int** a, int rows, int cols){
    connected_component_labeling_parallel_util(a, rows, cols, {0, 0}, {rows - 1, cols - 1});
}


int master_main(int argc, char* argv[]){
    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
    int show_images = 0,  method = 0;
    string run_type;
    void (*connected_component_labeling)(int**, int, int);
    string usage = "Usage:\t" + string(argv[0]) +" image_path [-hsv] [-show_images] [-tile_size=32]\n";
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
        else if(ar == "-h"){
            cout << usage;
            return 0;
        }
        else if(ar.rfind(tile) == 0){
            TILE_SIZE = stoi(ar.substr(tile.length()));
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


unordered_map<int, pair<int, int>> solve_tile(int* a, int rows, int cols){
    unordered_map<int, pair<int, int>> parent;
    for (int i = 0; i <  rows; ++i){
        for (int j = 0; j < cols; ++j){
            int idx = i * rows + j;
            if(!a[idx]){
                continue;
            }
            vector<int> neighbours;
            int m = numeric_limits<int>::max();
            for(auto d : directions){
                int ncol = j + d.second;
                int nrow = i + d.first;
                if ((ncol >= 0 && ncol < cols) && (nrow >= 0 && nrow < rows)){
                    int neighbour = a[idx];
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

int slave_main(int rank){
    MPI_Status status;
    MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int tag = status.MPI_TAG;
    while(tag != END_TAG){
        int msg_size = TILE_SIZE + 3;
        int* a = new int[msg_size];
        MPI_Recv(a, msg_size, MPI_INT, 0, tag, MPI_COMM_WORLD, &status); // rows, cols , a[TILE_SIZE]
        int rows = a[0];
        int cols = a[1];
        int start_i = a[2];
        int start_j = a[3];
        unordered_map<int, pair<int, int>> parent = solve_tile(a + 4, rows, cols);
        int parent_size = 3 * parent.size();
        int *par_sz= new int[parent_size];
        int i = 0;
        for(auto x : parent){
            par_sz[i] = x.first;
            par_sz[i + 1] = x.second.first;
            par_sz[i + 2] = x.second.second;
        }
        int all_size = 3 + TILE_SIZE + 3 * parent_size;
        int* all = new int[all_size];
        all[0] = start_i;
        all[1] = start_j;
        all[2] = parent_size;
        int* msg_to_send = all + 3;
        copy(a, a + TILE_SIZE, msg_to_send);
        copy(par_sz, par_sz + parent_size, msg_to_send + TILE_SIZE);
        MPI_Send(msg_to_send, all_size, MPI_INT, 0, LABEL_TAG, MPI_COMM_WORLD);  // a[TILE_SIZE], parent[TILE_SIZE], size[TILE_SIZE]
        delete [] a;
        delete [] all;
        delete [] par_sz;
    }
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