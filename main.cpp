// g++ main.cpp -o output `pkg-config --cflags --libs opencv`
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

using namespace std;
using namespace cv;


vector<pair<int, int>> directions {
                                    {-1, -1}, {-1, 0}, {-1, 1},
                                    {1, 1}, {1, 0}, {1, -1},
                                    {0, -1}, {0, 1}
                                  };


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

int** label_mat(const Mat& mat){
    int** labeled = new int*[mat.rows];
    for (int i = 0; i < mat.rows; ++i){
        labeled[i] = new int[mat.cols];
    }
    for(int row = 0; row < mat.rows; ++row) {
        auto* p = mat.ptr(row);
        for(int col = 0; col < mat.cols; ++col) {
            if(!p[col]){
                labeled[row][col] = row * col + col + 1;
            }
            else {
                labeled[row][col] = 0;
            }
        }
    }
    return labeled;
}

void connected_component_labeling(int** a, int rows, int cols){
    unordered_map<int, vector<int>> linked ;
    for (int i = 0; i <  rows; ++i){
        for (int j = 0; j < cols; ++j){
            vector<int> neighbours;
            int m = a[i][j];
            for(auto d : directions){
                int neighbour = a[i + d.first][j + d.second];
                if(neighbour){
                    neighbours.push_back(neighbour);
                    m = min(m, neighbour);
                }
            }
            if(neighbours.empty()){
                vector<int> v { a[i][j] };
                linked[a[i][j]] = v;
            }
            else {
                a[i][j] = m;
                for (int n : neighbours){
                    set_union(linked[n].begin(), linked[n].end(), neighbours.begin(), neighbours.end(),
                              back_inserter(linked[n]));
                }

            }

        }
    }

}


int main(int argc, char* argv[]) {
    Mat img = imread(argv[1]);
    int method = 0;
    if(argc > 2) {
        method = 1;
    }
    Mat out = transform_image_2_binary(img, method);
    imwrite("./black-white.jpg", out);

    int** labeled = label_mat(out);
//    for(auto i = 0; i < out.rows; ++i){
//        for(auto j = 0 ; j < out.cols; ++j){
//            cout << labeled[i][j] << " ";
//        }
//        cout << "\n";
//    }
//    cout << "\n\n\n";
//    cout<< labeled[0][0];
//    cout << out;



    // delete labeled matrix
    for (int i = 0; i < out.rows; ++i){
        delete [] labeled[i];
    }
    delete [] labeled;





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