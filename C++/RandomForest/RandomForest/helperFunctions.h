#include <iostream>
#include<fstream>
#include<cstring>
#include <Windows.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
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

int calcMaxIndex(cv::Mat& matrix, int numOfValues);

void removeFalsePredictions(cv::Mat& pred);

void calcClusters(cv::Mat& predictions,cv::Mat& visulizeClusters, cv::Mat& responses, int connectionThres, int imageWidth, int imageHeight, 
	int charSize, int fontSize, int minCluster,int overlapTileX, int overlapTileY);

void findCluster(cv::Mat& clusters, cv::Mat& pred, int x, int y, std::vector<int>& clusterSize, int clusterNum, int connectionThres);

std::vector<CvRTrees*> loadForestsOCR(int numOfForests, int maxDepth, int maxNumOfTreesInForest, int numOfChars, int numOfFalseImages, int tileSizeX, int tileSizeY, 
	 double angle, std::string charType, std::string featureType, bool falseClass, bool useNoise);

std::vector<CvRTrees*> loadForestsBackground();

int calcRectFiltNum(int charSizeX, int charSizeY);
void rotate(cv::Mat& src, double angle,int imageSizex, int imageSizey, double scale);
//RandomCharacters produceProxData(std::string type, int numOfClasses, int charSize, double fontSize);
void mouseCallback( int event, int x, int y, int flags, void* param );
void draw_box(cv::Mat * img, cv::Rect rect);
void writeSizeToFile(int imSizeX, int imSizeY, const char* filename);
CSize loadSizeFromFile(const char* filename);

std::string intToStrOCR(int i, int numOfChars,int numOfFalseImages,int tileSizeX, int tileSizeY, int depth, int treeNum, double angle, std::string charType, 
	std::string featureType,bool falseClass, bool n, bool useNoise);

std::string intToStrBackground(int i, int numOfTrueChars, int numOfFalseChars, int numOfFalseImages, int reSizeTo, std::string featureType, bool n);

void createAndSavePointPairs(int numOfPoints, int width, int height, std::string filename);
void preProcessRect(cv::Mat& image, double threshold);
void drawSquareToAdjustImage( int event, int x, int y, int flags, void* param );

std::vector<cv::Mat*> drawRandomImages(int numOfImages,int imageSize, int numOfLines, int numOfRectangles, bool useNoise);

void writeMatToFile(std::vector<std::vector<int>*> v, const char* filename);

std::string getImageAndGroundTruthName(int imageNum, char p, bool getGroundTrue);

int calcCharHeight(cv::Mat& im);

#endif