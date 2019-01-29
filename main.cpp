// g++ main.cpp -o output `pkg-config --cflags --libs opencv`
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>

using namespace std;
using namespace cv;


vector<pair<int, int>> directions {
                                    {-1, -1}, {-1, 0}, {-1, 1},
                                    {1, 1}, {1, 0}, {1, -1},
                                    {0, -1}, {0, 1}
                                  };


//union-find functions

int find_root(unordered_map<int, pair<int, int>>& parent, int x){
    if(parent[x].first != x){
        parent[x].first = find_root(parent, parent[x].first);
    }
    return parent[x].first;
}


void union_sets(unordered_map<int, pair<int, int>>& parent, int x, int y){
    int root1 = find_root(parent, x);
    int root2 = find_root(parent, y);
    if (root1 == root2){
        return;
    }
    if(parent[root1].second > parent[root2].second){
        parent[root2].first = root1;
        parent[root2].second += parent[root1].second;
    }
    else {
        parent[root1].first = root2;
        parent[root1].second += parent[root2].second;
    }
}

void make_set(unordered_map<int, pair<int, int>>& parent, int x){
    parent[x].first = x;
    parent[x].second = 1;

}

//


Mat transform_image_2_binary(const Mat& img, int method){
    Mat gray, blur, out;
    if (method){
        //HSV
        cvtColor(img, gray, COLOR_BGR2HSV);
        vector<Mat> channels;
        split(gray, channels);
        threshold(channels[1], out, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
    }
    else {
        //Grayscale
        cvtColor(img, gray, CV_BGR2GRAY);
        GaussianBlur(gray, blur,Size(5,5),0);
        threshold(blur, out, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
    }
    medianBlur(out, out, 5);
    return out;
}

int** convert_mat(const Mat& mat){
    int** data = new int*[mat.rows];
    for (int i = 0; i < mat.rows; ++i){
        data[i] = new int[mat.cols];
    }
    for(int row = 0; row < mat.rows; ++row) {
        auto* p = mat.ptr(row);
        for(int col = 0; col < mat.cols; ++col) {
            if(!p[col]){
                data[row][col] = 0;
            }
            else {
                data[row][col] = 1;
            }
        }
    }
    return data;
}

int** connected_component_labeling(int** a, int rows, int cols){
    unordered_map<int, pair<int, int>> parent;
    int** labeled = new int*[rows];
    for (int i = 0; i < rows; ++i){
        labeled[i] = new int[cols];
        fill(labeled[i], labeled[i] + cols, 0);
    }
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
                    int neighbour = labeled[nrow][ncol];
                    if(neighbour){
                        neighbours.push_back(neighbour);
                        m = min(m, neighbour);
                    }
                }
            }
            if(neighbours.empty()){
                labeled[i][j] = i * cols + j + 1;
                make_set(parent, labeled[i][j]);
            }
            else {
                labeled[i][j] = m;
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
            labeled[i][j] = find_root(parent, labeled[i][j]);
        }
    }
    return labeled;

}



int main(int argc, char* argv[]) {
    Mat img = imread(argv[1]);
    int method = 0;
    if(argc > 2) {
        method = 1;
    }
    Mat out = transform_image_2_binary(img, method);
    imwrite("./black-white.jpg", out);

    int** data = convert_mat(out);
    int** labeled = connected_component_labeling(data, out.rows, out.cols);
    for(int i = 0; i < out.rows; ++i){
        for(int j = 0 ; j < out.cols; ++j){
            cout << labeled[i][j] << " ";
        }
        cout << "\n";
    }
//    cout << "\n\n\n";
//    cout<< labeled[0][0];
//    cout << out;



    // delete labeled matrix
    for (int i = 0; i < out.rows; ++i){
        delete [] data[i];
    }
    delete [] data;





 //  adaptiveThreshold(blur, out, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 1); // ADAPTIVE_THRESH_GAUSSIAN_C , ADAPTIVE_THRESH_MEAN_C



//    adaptiveThreshold(blur, out, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 1);
//    double thresh_im = threshold(out, out, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
//    thresh = bitwise_or(adapt_thresh_im, thresh_im)




//    imwrite("./gray.jpg", gray);
//    imwrite("./blur.jpg", blur);

//    imshow("input", img);
//    imshow("intermediary", gray);
//    imshow("black-white", out);
//    waitKey(0);
    return 0;
}