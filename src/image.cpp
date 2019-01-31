

#include "../include/image.h"
#include "../include/utils.h"

cv::Mat transform_image_2_binary(const cv::Mat& img, int method){
    cv::Mat gray, blur, out;
    if (method){
        //HSV
        cvtColor(img, gray, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> channels;
        split(gray, channels);
        threshold(channels[1], out, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    }
    else {
        //Grayscale
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        GaussianBlur(gray, blur,cv::Size(5,5),0);
        threshold(blur, out, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    }
    medianBlur(out, out, 5);
    return out;
}

int** convert_mat(const cv::Mat& mat){
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


uchar* color_labels(int** a, int rows, int cols){
    std::unordered_map<int, int> color_map;
    int sz = rows * cols;
    uchar* colored = new uchar[sz * 3];
    std::fill(colored, colored + sz * 3, 0);
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


cv::Mat create_output_mat(uchar* data, int rows, int cols){
    cv::Mat final;
    cv::Mat channelR(rows, cols, CV_8UC1, data);
    cv::Mat channelG(rows, cols, CV_8UC1, data + rows * cols);
    cv::Mat channelB(rows, cols, CV_8UC1, data + 2 * rows * cols);
    std::vector<cv::Mat> channels{ channelB, channelG, channelR };
    merge(channels, final);
    return final;
}






