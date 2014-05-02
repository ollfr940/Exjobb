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


#ifndef FUNCTIONS_H
#define FUNCTIONS_H


//RandomCharacters produceData(int numOfChars, int charSize, std::string type,double angle, int charDivX, int charDivY, int charOrg, 
	//double fontSize, bool falseClass, bool useNoise);

void evaluateResult(std::vector<cv::Mat*> predictions,RandomCharactersImages& randIms, int imageWidth, int imageHeight, int charSizeX, int charSizeY,
	int numOfImages, int overlap, int minCluster, int pixelConnectionThres);

std::vector<cv::Mat*> predictImages(RandomCharactersImages& randIms, std::vector<CvRTrees*> forestVector1, std::vector<CvRTrees*> forestVector2,int imNum, int imageWidth, int imageHeight, int charSizeX, 
	int charSizeY, int overlap,int numOfTrees,double desicionThres1, double desicionThres2, int numOfPointPairs1, int numOfPointsPairs2, std::string charType, std::string featureType, int downSample, bool useNoise);

//RandomCharactersImages createTestImages(int numOfImages, int numOfChars, int charSize, int imageWidth,int imageHeight, std::string type,double angle, 
	//double fontSize, int numOfClasses);

RandomCharacters produceDataFromImage(std::vector<cv::Rect*> boxVec, std::vector<char> boxRes, int numOfCharacters, double angle, cv::Mat& image, bool useRealIm);

void evaluateIm(std::vector<CvRTrees*> forestVector, int testNum, int tileSizeX, int tileSizeY,std::string type, std::string featureType,int charDivX, int charDivY,
	int fontSize, std::string charType, int numOfPoints, double angle, int numOfTrees, double threshold,bool falseClass, bool useNoise);

RandomCharacters produceDataFromAfont(int numOfChars, std::string type, int numOfFalseData, int charDivX, int tileSizeX, int tileSizeY, int charDivY, double angle, bool falseClass
	,bool useNoise);

RandomCharactersImages createTestImagesAfont(int numOfImages, int numOfChars, int tileSizeX, int tileSizeY, int charDivX, int charDivY, int imageWidth, int imageHeight, 
	std::string type,double angle, std::string charType,bool useNoise);

void evaluateBackground(cv::Mat& image, std::vector<CvRTrees*> forestVector1, int tileSizeX, int tileSizeY, int numOfPointPairs1, double desicionThres2, int overlap);

void calcTreeForPlot(CvRTrees* forest, int numOfPoints,int tileSizeX, int tileSizeY, int charDivX, int charDivY, bool useNoise, int maxDepth);
#endif