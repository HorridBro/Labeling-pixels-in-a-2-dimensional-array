
#ifndef LABELING_PIXELS_IN_A_2_DIMENSIONAL_ARRAY_IMAGE_H
#define LABELING_PIXELS_IN_A_2_DIMENSIONAL_ARRAY_IMAGE_H

#include "includes.h"
cv::Mat transform_image_2_binary(const cv::Mat& img, int method);
int** convert_mat(const cv::Mat& mat);
uchar* color_labels(int** a, int rows, int cols);
cv::Mat create_output_mat(uchar* data, int rows, int cols);

#endif
