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

void writeMatToFile(cv::Mat& m,cv::Mat& r,int imageNum, const char* filename);

//RandomCharacters produceData(int numOfChars, int charSize, std::string type,double angle, int charDivX, int charDivY, int charOrg, 
	//double fontSize, bool falseClass, bool useNoise);

void evaluateResult(std::vector<cv::Mat*> predictions,RandomCharactersImages& randIms, int imageWidth, int imageHeight, int charSizeX, int charSizeY,
	int numOfImages, int overlap, int downSample, int minCluster, int pixelConnectionThres);

std::vector<cv::Mat*> predictImages(RandomCharactersImages& randIms, std::vector<CvRTrees*> forestVector,int imNum, int imageWidth, int imageHeight, int charSizeX, 
	int charSizeY, int overlap,int numOfTrees,double desicionThres, int numOfPointPairs, std::string charType, std::string featureType, int downSample, bool useNoise);

//RandomCharactersImages createTestImages(int numOfImages, int numOfChars, int charSize, int imageWidth,int imageHeight, std::string type,double angle, 
	//double fontSize, int numOfClasses);

RandomCharacters produceDataFromImage(std::vector<cv::Rect*> boxVec, std::vector<char> boxRes, int numOfCharacters, double angle, cv::Mat& image, bool useRealIm);

void evaluateIm(std::vector<CvRTrees*> forestVector, int testNum, int imageSize,std::string type, std::string featureType,int charDivX, int charDivY, int charOrg,
	int fontSize, std::string charType, int numOfPoints, double angle, int numOfTrees, double threshold,bool falseClass, bool useAfont, int downSample, bool useNoise);

RandomCharacters produceDataFromAfont(int numOfChars, std::string type, int numOfFalseData, int charDivX, int charDivY, double angle, bool falseClass
	, int downSample,bool useNoise);

RandomCharactersImages createTestImagesAfont(int numOfImages, int numOfChars, int charSize, int charDivX, int charDivY, int imageWidth, int imageHeight, 
	std::string type,double angle, double fontSize, std::string charType, int downSample,bool useNoise);

#endif