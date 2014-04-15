#include <iostream>
#include<fstream>
#include"Classes.h"
#include"helperFunctions.h"
#include<cstring>
#include<Windows.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
#include <vector>
#include <math.h>
#include<cmath>

#ifndef FEATURES_H
#define FEATURES_H

void calcRectFeatureTile(cv::Mat& tile, cv::Mat& featureMat, int width, int height, int im);
cv::Mat calcFeaturesTraining(RandomCharacters trainingData, int numOfPoints, std::string featureType, float downSample, bool useNoise);
void calcPointPairsFeaturesTile(cv::Mat& tile, cv::Mat& featureMat, cv::Mat& pointVector, int numOfPoints, int im, bool useNoise);
cv::Mat calcPointPairsFeaturesTraining(RandomCharacters trainingData, int numOfPoints);

#endif