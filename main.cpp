// g++ main.cpp -o output `pkg-config --cflags --libs opencv`
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <chrono>


using namespace std;
using namespace cv;
using namespace std::chrono;


// check all 8 neighbours
//vector<pair<int, int>> directions {
//                                    {-1, -1}, {-1, 0}, {-1, 1},
//                                    {1, 1}, {1, 0}, {1, -1},
//                                    {0, -1}, {0, 1}
//                                  };

//check only w, n-w, n ,e neighbours
vector<pair<int, int>> directions { {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };

vector<int> colors = {
        0x00FF00,
        0x0000FF,
        0xFF0000,
        0x01FFFE,
        0xFFA6FE,
        0xFFDB66,
        0x006401,
        0x010067,
        0x95003A,
        0x007DB5,
        0xFF00F6,
        0xFFEEE8,
        0x774D00,
        0x90FB92,
        0x0076FF,
        0xD5FF00,
        0xFF937E,
        0x6A826C,
        0xFF029D,
        0xFE8900,
        0x7A4782,
        0x7E2DD2,
        0x85A900,
        0xFF0056,
        0xA42400,
        0x00AE7E,
        0x683D3B,
        0xBDC6FF,
        0x263400,
        0xBDD393,
        0x00B917,
        0x9E008E,
        0x001544,
        0xC28C9F,
        0xFF74A3,
        0x01D0FF,
        0x004754,
        0xE56FFE,
        0x788231,
        0x0E4CA1,
        0x91D0CB,
        0xBE9970,
        0x968AE8,
        0xBB8800,
        0x43002C,
        0xDEFF74,
        0x00FFC6,
        0xFFE502,
        0x620E00,
        0x008F9C,
        0x98FF52,
        0x7544B1,
        0xB500FF,
        0x00FF78,
        0xFF6E41,
        0x005F39,
        0x6B6882,
        0x5FAD4E,
        0xA75740,
        0xA5FFD2,
        0xFFB167,
        0x009BFF,
        0xE85EBE,
};

// utility functions

void delete_matrix(int** data, int rows){
    for (int i = 0; i < rows; ++i){
        delete [] data[i];
    }
    delete [] data;
}

//

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

// main functions

Mat transform_image_2_binary(const Mat& img, int method){
    Mat gray, blur, out;
    if (method){
        //HSV
        cvtColor(img, gray, COLOR_BGR2HSV);
        vector<Mat> channels;
        split(gray, channels);
        threshold(channels[1], out, 0, 255, THRESH_BINARY + THRESH_OTSU);
    }
    else {
        //Grayscale
        cvtColor(img, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blur,Size(5,5),0);
        threshold(blur, out, 0, 255, THRESH_BINARY + THRESH_OTSU);
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
                data[row][col] = row* mat.rows + col + 1;
            }
        }
    }
    return data;
}


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


uchar* color_labels(int** a, int rows, int cols){
    unordered_map<int, int> color_map;
    int sz = rows * cols;
    uchar* colored = new uchar[sz * 3];
    fill(colored, colored + sz * 3, 0);
    int id = 0;
    for(int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int elem = a[i][j];
            if(!elem){
                continue;
            }
            int current_color;
            if(color_map.find(elem) == color_map.end()){
                current_color = colors[id % colors.size()];
                id++;
                color_map[elem] = current_color;
            }
            else{
                current_color = color_map[elem];
            }
            uchar r = (uchar)((current_color >> 16) & 0xFF);
            uchar g = (uchar)((current_color >> 8) & 0xFF);
            uchar b = (uchar)((current_color) & 0xFF);
            int index = i * cols + j;
            colored[index] = r;
            colored[sz + index] = g;
            colored[sz * 2 + index] = b;
        }
    }
    return colored;
}


Mat create_output_mat(uchar* data, int rows, int cols){
    Mat final;
    Mat channelR(rows, cols, CV_8UC1, data);
    Mat channelG(rows, cols, CV_8UC1, data + rows * cols);
    Mat channelB(rows, cols, CV_8UC1, data + 2 * rows * cols);
    vector<Mat> channels{ channelB, channelG, channelR };
    merge(channels, final);
    return final;
}


int main(int argc, char* argv[]) {
    int show_images = 0,  method = 0;
    string usage = "Usage:\t" + string(argv[0]) +" image_path [-hsv] [-show_images]\n";
    if (argc < 2){
        cout << "Error, invalid args! " << usage;
        return 1;
    }
    for(int i = 1; i < argc; ++i){
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
    Mat img = imread(argv[1]);
    Mat out = transform_image_2_binary(img, method);
    int rows = out.rows , cols = out.cols;
    int** data = convert_mat(out);
    high_resolution_clock::time_point start = high_resolution_clock::now();
    connected_component_labeling_serial(data, rows, cols);
    high_resolution_clock::time_point stop = high_resolution_clock::now();
    float duration = duration_cast<chrono::duration<float>>( stop - start ).count();
    cout << "Connected Component Labeling Duration = " << duration << " seconds\n";
    uchar* colored = color_labels(data, rows, cols);
    Mat final = create_output_mat(colored, rows, cols);
    imwrite("./black-white.jpg", out);
    imwrite("./output.jpg", final);
    if (show_images){
        imshow("Input", img);
        imshow("Intermediary", out);
        imshow("Output", final);
        while(waitKey(1) != 27);
        img.release();
        out.release();
        final.release();
    }
    delete [] colored;
    delete_matrix(data, rows);
    return 0;
}