#include <iostream>
#include "Classes.h"
#include<fstream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
#include <vector>
#include <math.h>
#include <time.h>


#ifndef FUNCTIONS_H
#define FUNCTIONS_H

void writeMatToFile(cv::Mat& m,cv::Mat& r,int imageNum, const char* filename);
RandomCharacters produceData(int numOfChars, int charSize, std::string type);
cv::Mat createResponses(int trainingNum, int characters);
cv::Mat createRectFeatures(RandomCharacters trainingData, int numOfChars, int imSize);
void rotate(cv::Mat& src, double angle,int imageSize, double scale);
void evaluateRect(CvRTrees& tree, int testNum, int imageSize,std::string type);
void evaluateResult(std::vector<cv::Mat*> predictions,RandomCharactersImages& randIms, int imageSize, int charSize, int tileNum, int overlap);
std::vector<cv::Mat*> predictImages(RandomCharactersImages& randIms, CvRTrees& tree,int imNum, int imageSize, int charSize, int overlap, int tileNum, std::string type);
RandomCharactersImages createTestImages(int numOfImages, int numOfChars, int charSize, int imageSize, std::string type);

#endif