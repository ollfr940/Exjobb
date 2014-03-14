#include <iostream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

std::vector<cv::Mat*> produceTrainingData(int num, int imageSize);
cv::Mat createRectFeatures(std::vector<cv::Mat*> trainingData, int trainingNum, int imSize);

#endif