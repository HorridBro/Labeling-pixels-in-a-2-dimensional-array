//mpirun  -np 1 cmake-build-debug/main images/gray/aloi.jpg -show_images //serial
//mpirun  -np 8 cmake-build-debug/main images/gray/aloi.jpg -show_images //parallel

#include "include/includes.h"
#include "include/utils.h"
#include "include/union_find.h"
#include "include/image.h"
#include <mpi.h>

#include <thread>

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
                    if(global_parent.find(n) == global_parent.end()){
                        make_set(global_parent, n);
                    }
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
//        cout << "(parent  " << parent[i] << " " << parent[i+1] << " " << parent[i+2] << ")";
        global_parent[parent[i]] = {parent[i+1], parent[i+2]};
    }
//    cout << " \n";
}

void merge_parent_inv(unordered_map<int, pair<int, int>> & parent, int* par){
    int i = 1;
    par[0] = par[parent.size()];
    for(auto x : parent){
        par[i] = x.first;
        par[i + 1] = x.second.first;
        par[i + 2] = x.second.second;
        i += 3;
    }
}


int connected_component_labeling_parallel_util(int** a, int rows, int cols, pair<int, int> start, pair<int, int> stop){
    int size = rows * cols;
//    cout << rows << " "  << cols  << " cords\t" << start.first  << " "  << start.second << " " << stop.first << " " << stop.second<< endl;
    if(size == 1){
//        make_set(global_parent, a[start.first][start.second]);
        return 0;
    }
    if(size < TILE_SIZE){
        merge_tiles(a, start, stop);
        return 0;
    }
    if(size <= TILE_SIZE){
//        cout << "RIP\n";
        int msg_to_send [size + 4];
        MPI_Request req;
        int *temp_array = msg_to_send + 4;
        msg_to_send[0] = rows;
        msg_to_send[1] = cols;
        msg_to_send[2] = start.first;
        msg_to_send[3] = start.second;
//        cout << "to send\n";
        for(int i = start.first, ii = 0; i <= stop.first; i++, ii++){
            for(int j = start.second, jj = 0; j <= stop.second; j++, jj++){
                temp_array[ii * cols + jj] = a[i][j];
//                cout << a[i][j] << " ";
            }
//            cout << endl;
        }
        //TODO cu isend
        //MPI_Send(msg_to_send, size + 4, MPI_INT, dest, LABEL_TAG, MPI_COMM_WORLD);
        MPI_Isend(msg_to_send, size + 4, MPI_INT, dest, LABEL_TAG, MPI_COMM_WORLD, &req);
        cout << "DST" << dest << endl;
        dest = (dest + 1) % (numprocs -1) + 1;
        return 1;
    }
    else{

        int new_rows = rows / 2 - 1;
        int new_cols = cols / 2 - 1;
        int rows_left = new_rows + rows % 2;
        int cols_left = new_cols + cols % 2;
        int n_sends = 0;
        n_sends += connected_component_labeling_parallel_util(a, new_rows + 1 , new_cols + 1 , start , {start.first + new_rows, start.second + new_cols});
        n_sends += connected_component_labeling_parallel_util(a, new_rows + 1, cols_left + 1, {start.first, start.second + new_cols + 1}, {start.first + new_rows, start.second + new_cols + cols_left + 1});
        n_sends += connected_component_labeling_parallel_util(a, rows_left + 1, new_cols + 1, {start.first + new_rows + 1, start.second}, {start.first + new_rows + rows_left + 1, start.second + new_cols});
        n_sends += connected_component_labeling_parallel_util(a, rows_left + 1 , cols_left + 1, {start.first + new_rows + 1, start.second + new_cols + 1} , stop);
        if(size / 4 <= TILE_SIZE){
            int recv_size = 3 + TILE_SIZE * 6;
//            int* tiles = new int[recv_size * 4];
//            cout << "DADA\n";
            for(int i = 0 ; i < n_sends; i++){
                int tile [recv_size], r, c;
                MPI_Recv(tile, recv_size, MPI_INT, MPI_ANY_SOURCE, LABEL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int* tl = tile + 5;
                r = tile[3];
                c = tile[4];
//              cout << r << " " << c << " start  " << tile[0] << " " << tile[1] << " " << tile[0] + r -1 << " "<< tile[1] + c -1 <<endl;
                for(int k = tile[0], kk = 0; kk < r ; k++, kk++){
                    for(int l = tile[1], ll = 0; ll < c; l++, ll++){
                        a[k][l] = tl[kk * c + ll];
//                       cout<< tl[kk * c + ll] << " ";
                    }
//                    cout << "\n";
                }
                merge_parent(tl + r * c, tile[2]);

            }
        }
        merge_tiles(a, {start.first, start.second + new_cols}, {start.first + new_rows, start.second + new_cols + 1}); //top-left and top-rght
        merge_tiles(a, {start.first + new_rows + 1, start.second + new_cols}, {stop.first, start.second + new_cols + 1 }); //down-left and down-right
        merge_tiles(a, {start.first + new_rows, start.second}, {start.first + new_rows + 1, stop.second}); // top merged and down merged
    }

    return 0;


}



void connected_component_labeling_scatter(int* a, int rows, int cols){
    int rows_left = rows % numprocs;
    int size = rows * cols;
    int pieces = rows / numprocs;
    int piece_size = pieces * cols;
    int* initial_data = new int[numprocs * 2 + 2];
    int* sendcount = initial_data + 2;
    int* disps = initial_data + numprocs + 2;
    int* buff = new int[size];
    int* parent_serialized;
    initial_data[0] = pieces;
    initial_data[1] = cols;
    fill(sendcount, sendcount + numprocs, piece_size);
    disps[0] = 0;
    sendcount[0] += rows_left * cols;
    int skip = sendcount[0];
    for(int i = 1; i < numprocs; i++){
        disps[i] = skip;
        skip += sendcount[i];
    }
//    for(int i = 0; i < numprocs; i++){
//        cout << disps[i] << "  " << sendcount[i] <<"\n";
//    }

    MPI_Bcast(initial_data, numprocs * 2 + 2, MPI_INT, 0, MPI_COMM_WORLD);
    parent_serialized = new int[size]; // possibly too much


//    cout << "piece size" << piece_size <<"\n";

   MPI_Scatterv(a, sendcount, disps, MPI_INT, buff, sendcount[0], MPI_INT, 0, MPI_COMM_WORLD);
//   MPI_Scatter(a, piece_size, MPI_INT, a, piece_size, MPI_INT, 0, MPI_COMM_WORLD);

   global_parent = solve_tile(a, rows_left + pieces , cols);



//    MPI_Gather(bf, piece_size, MPI_INT, a, piece_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(a, sendcount[0], MPI_INT, a, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);



    MPI_Gatherv(parent_serialized, sendcount[0], MPI_INT, parent_serialized, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);





    for(int i = piece_size + rows_left * cols ; i < size; i += piece_size){
//        cout << "PAR " << parent_serialized[i] << " " << parent_serialized[i+1] << " " << parent_serialized[i+2]<< "\n";
         merge_parent(parent_serialized + i + 1, parent_serialized[i]);
    }
//    global_parent.erase(0);

//    cout << "\nhalp\n";
//
    for(int i = pieces + rows_left; i < rows ; i+= pieces){
//        cout << "i=" << i << "\n";
        merge_tiles_arr(a, {i - 1, 0}, {i, cols - 1}, cols);
    }
//    for(auto x : global_parent){
//        cout << x.first << " " << x.second.first << " " << x.second.second << endl;
//    }




    //MPI_Bcast(parent_serialized, size, MPI_INT, 0, MPI_COMM_WORLD);

    // merge parent


    delete [] initial_data;
    delete [] buff;




}
void connected_component_labeling_parallel(int* a, int rows, int cols) {

    connected_component_labeling_scatter(a, rows, cols);

//    connected_component_labeling_parallel_util(a, rows, cols, {0, 0}, {rows - 1, cols - 1});
//    MPI_Request req;
//    int parent_size = 3 * global_parent.size() + 1;

//    for(int i = 1; i < numprocs; i++){
//        MPI_Isend(&parent_size, 1, MPI_INT, i, END_TAG, MPI_COMM_WORLD, &req);
//
//    }
//    int *par_sz= new int[parent_size];
//    par_sz[1] = parent_size;
//    int i = 0;
//    for(auto x : global_parent){
//        par_sz[i] = x.first;
//        par_sz[i + 1] = x.second.first;
//        par_sz[i + 2] = x.second.second;
//        i += 3;
//    }
//
//    MPI_Bcast(par_sz, parent_size + 1, MPI_INT, 0, MPI_COMM_WORLD);
//    MPI_SCATT

//    for(auto dd : global_parent){
//        cout << dd.first << " ("  << dd.second.first << " " << dd.second.second << ") " ;
//    }
//    cout << endl;
    //parallel with scatter
    for (int i = 0 ; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            int idx = i * cols + j;
            if(!a[idx]){
                continue;
            }
           if(global_parent.find(a[idx]) != global_parent.end()){
                a[idx] = find_root(global_parent, a[idx]);
            }
        }
    }
//    delete [] par_sz;

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
//        else if(ar == "-h"){
//            cout << usage;
//            return 0;
//        }
//        else if(ar.rfind(tile) == 0){
//            TILE_SIZE = stoi(ar.substr(tile.length()));
//        }
    }
    if(numprocs > 1){
//        connected_component_labeling = connected_component_labeling_parallel;
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
    int *data_array = new int[rows * cols];
    for(int i = 0 ; i < rows; i++){
        for(int j = 0 ; j < cols ; j++){
            data_array[i * cols + j] = data[i][j];
        }
    }

     //testing
//    rows = 21;
//    cols = 5;
//    int *data1 = new int[rows * cols];
////    for (int i = 0; i < rows; ++i){
////        data[i] = new int[cols];
//    srand(time(0));
////    }
//    for(int i = 0; i< rows; i++){
//        for(int j = 0; j< cols; j++){
//            data1[i * cols + j] = 0;
//            if(i  % 4){
//                data1[i * cols + j] = i*cols + j + 1;
//            } else{
//                data1[i * cols + j] = 0;
//            }
//        }
//    }
//    data_array = data1;
//    for(int i = 0; i< rows; i++){
//        for(int j = 0; j< cols; j++){
//            cout << data_array[i * cols + j] << "\t";
//        }
//        cout << endl;
//    }
//    cout << "\n\n\n";

    //
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

//   this_thread::sleep_for( (chrono::seconds(2)));
//    cout << "\n\n\n";
//
//    for(int i = 0; i< rows; i++){
//        for(int j = 0; j< cols; j++){
//            cout << data_array[i *cols+j] << "\t";
//        }
//        cout << endl;
//    }
//    return 0;
//    exit(0);

    cout << run_type << " Connected Component Labeling Duration = " << duration << " seconds\n";
    uchar* colored = color_labels(data, rows, cols);
    Mat final = create_output_mat(colored, rows, cols);
    imwrite("./black-white" + run_type + ".jpg", out);
    imwrite("./output"  + run_type + ".jpg", final);
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
//    cout << "enter\n\n";
    int *initial_data, rows, cols, size, i = 1, parent_size, *parent_serialized, *a, *b, *par_buff, *sendcount, *disps;
    initial_data = new int [numprocs * 2 + 2];
    MPI_Bcast(initial_data, numprocs * 2 + 2, MPI_INT, 0, MPI_COMM_WORLD);
    rows = initial_data[0];
    cols = initial_data[1];
    sendcount = initial_data + 2;
    disps = initial_data + numprocs + 2;
    size = rows * cols;
    a = new int[size];
    b = new int[size];
//    cout << "sfafaRECV  "<<  rows << " " << cols<< " "<< size << "\n";
    MPI_Scatterv(b, sendcount, disps, MPI_INT, a, size, MPI_INT, 0, MPI_COMM_WORLD);
//    cout << "dadad\n";
    unordered_map<int, pair<int, int>> parent = solve_tile(a, rows, cols);
//    if(rank ==1){
//    for(int ii = 0; ii< rows; ii++){
//        for(int jj = 0; jj< cols; jj++){
//            int idx = ii * cols + jj;
//            cout << a[idx] << "\t";
//        }
//        cout << endl;
//    }
//    }

    int par_sz = int(parent.size() * 3);
    parent_size = max(par_sz, size);
    parent_serialized = new int[parent_size];
    parent_serialized[0] = par_sz;
    for(auto x : parent){
        parent_serialized[i] = x.first;
        parent_serialized[i + 1] = x.second.first;
        parent_serialized[i + 2] = x.second.second;
//        cout << "PAR " << parent_serialized[i] << " " << parent_serialized[i+1] << " " << parent_serialized[i+2]<< "\n";
        i += 3;
    }
//    parent_serialized[parent_size - 1] = 0;
//   cout << "scattera " << rank << "\n";

//    MPI_Gather(a, size, MPI_INT, b, 0, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(a, size, MPI_INT, b, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);
//    cout << "scatter pare " << rank <<"\n";
    //MPI_Gather(parent_serialized, size, MPI_INT, b, 0, MPI_INT, 0, MPI_COMM_WORLD);

//    cout << "BFG "<< size << " " << parent_size << " " << par_sz << "\n";
    MPI_Gatherv(parent_serialized, size, MPI_INT, b, sendcount, disps, MPI_INT, 0, MPI_COMM_WORLD);
//    cout << "AFG\n";
//    cout << "DONE " << rank <<"\n";
//    int total_size = size * numprocs;
//    int* all_parents = new int[total_size];
//    MPI_Bcast(all_parents, size, MPI_INT, 0, MPI_COMM_WORLD);
    delete [] a;
    delete [] b;
    delete [] initial_data;
    delete [] parent_serialized;
    return 0;












    MPI_Status status;
    MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int tag = status.MPI_TAG;
    while(tag != END_TAG){
        int msg_size = 2 * TILE_SIZE + 4;
        int* a = new int[msg_size];
//        cout << "PREP RECV\n";
        MPI_Recv(a, msg_size, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // rows, cols , a[TILE_SIZE]
//        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
//        tag = status.MPI_TAG;
//        if(status.MPI_TAG == END_TAG){
//            break;
//        }
        int rows = a[0];
        int cols = a[1];
        int start_i = a[2];
        int start_j = a[3];
        int *b = a + 4;
        TILE_SIZE = rows * cols;
//        cout << "SLAVE " << rows << " " << cols << "\t" << start_i << " " << start_j  << endl;
//        for(int ii = 0 ; ii < rows; ii++){
//            for(int j= 0 ; j < cols; j++){
//                cout << b[ii * cols + j] << " ";
//            }
//            cout << endl;
//        }
        unordered_map<int, pair<int, int>> parent = solve_tile(b, rows, cols);
        int parent_size = 3 * parent.size();
        int *par_sz= new int[parent_size];
        int i = 0;
        for(auto x : parent){
            par_sz[i] = x.first;
            par_sz[i + 1] = x.second.first;
            par_sz[i + 2] = x.second.second;
//            cout << "x " << x.first << " " << x.second.first << " " << x.second.second;
            i += 3;
        }
//        cout << "i= " << i << " "  << parent_size << endl;
//        cout << "\nSLAVE AFTER " << rows << " " << cols << "\t" << start_i << " " << start_j  << endl;
//        for(int ii = 0 ; ii < rows; ii++){
//            for(int j= 0 ; j < cols; j++){
//                cout << b[ii * cols + j] << " ";
//            }
//         cout << endl;
//        }
        int all_size = 5 + TILE_SIZE  + parent_size;
        int* all = new int[all_size];
        all[0] = start_i;
        all[1] = start_j;
        all[2] = parent_size;
        all[3] = rows;
        all[4] = cols;
        int* msg_to_send = all + 5;
        copy(b, b + TILE_SIZE, msg_to_send);
        copy(par_sz, par_sz + parent_size, msg_to_send + TILE_SIZE);
//        cout << "\n\nsz"  <<  parent_size << " " << all_size;
//        cout << "msg_to_snd\n";
//        for(int i = 0 ;i < rows; i++){
//            for( int j = 0 ; j < cols; j++){
//            cout << msg_to_send[i * cols + j] << " " ;
//            }
//            cout << " \n ";
//        }
        MPI_Send(all, all_size, MPI_INT, 0, LABEL_TAG, MPI_COMM_WORLD);  // a[TILE_SIZE], parent[TILE_SIZE], size[TILE_SIZE]
//        cout << "#####send " << start_i << "\t" << start_j << " sz = "<<TILE_SIZE << endl;
        delete [] a;
        delete [] all;
        delete [] par_sz;
    }
    cout << "END id=" << rank << "\n";
//    int recv_sz[1], sz;
//    MPI_Recv(recv_sz, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
//    sz = recv_sz[0];
//    unordered_map<int, pair<int, int>> par;
//    int* recv = new int[sz];
//    MPI_Bcast(recv, sz, MPI_INT, 0, MPI_COMM_WORLD);
//    for(int i = 0 ; i < sz; i +=3 ){
//        par[recv[i]].first = recv[i+1];
//        par[recv[i]].second = recv[i+2];
//    }
//    for (int i = 0 ; i < rows; ++i){
//        for (int j = 0; j < cols; ++j){
//            if(!a[i][j]){
//                continue;
//            }
//            if(par.find(a[i][j]) != global_parent.end()){
//                a[i][j] = find_root(global_parent, a[i][j]);
//            }
//        }
//    }



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