#include <iostream>
#include "Classes.h"
#include "features.h"
#include<fstream>
#include<cstring>
#include<Windows.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
#include <vector>
#include <math.h>
#include <time.h>

#ifndef FUNCTIONSFORREALIMAGES
#define FUNCTIONSFORREALIMAGES

void evaluateResultRealImage(std::vector<cv::Mat*> predictions,std::vector<cv::Mat*> imageVector, int imageWidth, int imageHeight, int charSizeX, int charSizeY,
	int numOfImages, int overlap, float downSample);

std::vector<cv::Mat*> predictRealImages(std::vector<cv::Mat*> imageVector, std::vector<CvRTrees*> forestVector,int imNum, int imageWidth, int imageHeight, int charSizeX, 
	int charSizeY, int overlap,int numOfTrees,double desicionThres, int numOfPointPairs, std::string charType, std::string featureType, float downSample, bool useNoise);

#endif