#include <iostream>
#include "Classes.h"
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
extern  bool destroy;
extern  cv::Rect box;
extern  bool drawing_box;
extern std::vector<cv::Rect*> boxVector;
extern std::vector<char> boxResponses;
extern bool firstBox;
extern int firstBoxWidth;
extern int firstBoxHeight;

void writeMatToFile(cv::Mat& m,cv::Mat& r,int imageNum, const char* filename);
RandomCharacters produceData(int numOfChars, int charSize, std::string type,double angle, double fontSize, int numOfClasses);
//cv::Mat createResponses(int trainingNum, int characters);
cv::Mat createRectFeatures(RandomCharacters trainingData);
void rotate(cv::Mat& src, double angle,int imageSizex, int imageSizey, double scale);
//void evaluateRect(CvRTrees& tree, int testNum, int imageSize,std::string type, double angle);
void evaluateResult(std::vector<cv::Mat*> predictions,RandomCharactersImages& randIms, int imageSize, int charSizeX, int charSizeY, int numOfImages, int overlap);
std::vector<cv::Mat*> predictImages(RandomCharactersImages& randIms, std::vector<CvRTrees*> forestVector,int imNum, int imageSizeX, int imageSizeY, int charSize, int overlap,int numOfTrees,double desicionThres, std::string type, cv::Mat& proxDataFeatures);
RandomCharactersImages createTestImages(int numOfImages, int numOfChars, int charSize, int imageSize, std::string type,double angle, double fontSize, int numOfClasses);
RandomCharacters produceProxData(std::string type, int numOfClasses, int charSize, double fontSize);
int calcRectFiltNum(int charSizeX, int charSizeY);
void mouseCallback( int event, int x, int y, int flags, void* param );
void draw_box(cv::Mat * img, cv::Rect rect);
RandomCharacters produceDataFromImage(std::vector<cv::Rect*> boxVec, std::vector<char> boxRes, int numOfCharacters, double angle, cv::Mat& image);
void writeSizeToFile(int imSizeX, int imSizeY, const char* filename);
CSize loadSizeFromFile(const char* filename);
std::string intToStr(int i, int numOfChars,int numOfClasses, std::string type);

#endif