#include <iostream>
#include<fstream>
#include<cstring>
#include <Windows.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Classes.h"

#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H

extern  bool destroy;
extern  cv::Rect box;
extern  bool drawing_box;
extern std::vector<cv::Rect*> boxVector;
extern std::vector<char> boxResponses;
extern bool firstBox;
extern int firstBoxWidth;
extern int firstBoxHeight;

int calcRectFiltNum(int charSizeX, int charSizeY);
void rotate(cv::Mat& src, double angle,int imageSizex, int imageSizey, double scale);
RandomCharacters produceProxData(std::string type, int numOfClasses, int charSize, double fontSize);
void mouseCallback( int event, int x, int y, int flags, void* param );
void draw_box(cv::Mat * img, cv::Rect rect);
void writeSizeToFile(int imSizeX, int imSizeY, const char* filename);
CSize loadSizeFromFile(const char* filename);
std::string intToStr(int i, int numOfChars,int numOfClasses, int depth, int treeNum, int angle, std::string charType, std::string featureType, bool n);
void preProcessRect(cv::Mat& image, double threshold);

#endif