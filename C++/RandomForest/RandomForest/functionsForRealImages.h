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

void calcClustersRealImage(cv::Mat& predictions,cv::Mat& visulizeClusters, int connectionThres, int imageWidth, int imageHeight, int tileSizeX, int tileSizeY, int fontSize,
	int minCluster,int overlap, bool useGroundTruth, int imageNum);

void evaluateResultRealImage(std::vector<cv::Mat*> predictions,std::vector<cv::Mat*> imageVector, int imageWidth, int imageHeight, int charSizeX, int charSizeY,
	int numOfImages, int overlap, int minCluster, int pixelConnectionThres, bool useGroundTruth, int imageNumber);

std::vector<cv::Mat*> predictRealImages(std::vector<cv::Mat*> imageVector, std::vector<CvRTrees*> forestVector1, std::vector<CvRTrees*> forestVector2,int imNum, int imageWidth, int imageHeight, int tileSizeX, 
	int tileSizeY, int overlap,int numOfTrees,double desicionThres1,double desicionThres2, int numOfPointPairs1, int numOfPointPairs2, int numOfPointPairs3, std::string charType, std::string featureType, bool useNoise);

void findClusters(cv::Mat& clusters, cv::Mat& pred, int x, int y, std::vector<int>& clusterSize, int clusterNum, int connectionThres);
#endif