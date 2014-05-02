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
cv::Mat calcFeaturesTraining(RandomCharacters trainingData, int numOfPoints, std::string featureType, int tileSizeX, int tileSizeY, bool useNoise);
void calcPointPairsFeaturesTile(cv::Mat& tile, cv::Mat& featureMat, cv::Mat& pointVector, int numOfPoints, int im, bool useNoise);
//cv::Mat calcPointPairsFeaturesTraining(RandomCharacters trainingData, int numOfPoints, bool useDiff);

RandomImagesAndCharacters calcPointPairFeaturesScales(std::vector<cv::Mat*> randomImages, RandomCharacters trainingData, int tileSizeX, int tileSizeY, int resizeTo,
	int imageSize,int numOfPointPairs, int numOfTrueCharacters, int numOfFalseCharacters, std::string typeOfChars, bool useNoise);

RandomImagesAndCharacters calcStandardDeviationFeatures(std::vector<cv::Mat*> randomImages, RandomCharacters trainingData, int tileSizeX, int tileSizeY, int imageSize, 
	int numOfTrueCharacters, int numOfFalseCharacters, std::string typeOfChars, int reSizeTo);

void calcStdTile(cv::Mat& tile, cv::Mat& featureMat, int im, int reSizeTo);

#endif