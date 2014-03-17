#include <iostream>
#include<fstream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
#include <vector>

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

void writeMatToFile(cv::Mat& m,cv::Mat& r,int imageNum, const char* filename);
std::vector<cv::Mat*> produceData(int first, int characters, int num, int imageSize);
cv::Mat createResponses(int trainingNum, int characters);
cv::Mat createRectFeatures(std::vector<cv::Mat*> trainingData, int trainingNum, int imSize);
cv::Mat createRectFeaturesTest(std::vector<cv::Mat*> trainingData, int trainingNum, int imSize);
cv::Mat creatSumFeatures(std::vector<cv::Mat*> trainingData, int trainingNum, int imageSize, bool test);

#endif