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
RandomCharacters produceData(int numOfChars, int charSize, std::string type,double angle, double fontSize, int numOfClasses);
void evaluateResult(std::vector<cv::Mat*> predictions,RandomCharactersImages& randIms, int imageSize, int charSizeX, int charSizeY, int numOfImages, int overlap);
std::vector<cv::Mat*> predictImages(RandomCharactersImages& randIms, std::vector<CvRTrees*> forestVector,int imNum, int imageSizeX, int imageSizeY, int charSize, int overlap,int numOfTrees,double desicionThres, int numOfPointPairs, std::string charType, std::string featureType);
std::vector<cv::Mat*> predictImagesRandomPoints(RandomCharactersImages& randIms, std::vector<CvRTrees*> forestVector,int imNum, int imageSizeX, int imageSizeY, int charSize, int overlap,int numOfTrees,double desicionThres, std::string type, int numOfPointPairs);
RandomCharactersImages createTestImages(int numOfImages, int numOfChars, int charSize, int imageSize, std::string type,double angle, double fontSize, int numOfClasses);
RandomCharacters produceDataFromImage(std::vector<cv::Rect*> boxVec, std::vector<char> boxRes, int numOfCharacters, double angle, cv::Mat& image);

#endif