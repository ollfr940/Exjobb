#include "stdafx.h"
#include "functionsForRealImages.h"
using namespace std;
using namespace cv;

vector<Mat*> predictRealImages(vector<Mat*> imageVector, vector<CvRTrees*> forestVector,int imNum, int imageWidth, int imageHeight, int charSizeX, int charSizeY, int overlap, 
	int numOfTrees,double desicionThres, int numOfPointPairs, string charType, string featureType, float downSample, bool useNoise)
{
	printf("Detecting characters in images....\n\n");
	//CalcRectSample calcRect;
	DWORD start, stop;
	int xPos, yPos, predPosx, predPosy;
	int overlapTileX = charSizeX/overlap;
	int overlapTileY = charSizeY/overlap;

	Mat imRect, imRectThres, integralRect, featureMat;
	int imRectUp,imRectDown, imRectMiddleHor, imRectMiddleVert, rectFiltNum, imRectSum;
	vector<Mat*> predictions;
	Mat* pred;
	char proxIndx;
	int tileNumX = imageWidth/(charSizeX/overlap) - 2;//imageWidth/charSizeX*overlap - (overlap-1);
	int tileNumY = imageHeight/(charSizeY/overlap) - 2; //imageHeight/charSizeY*overlap -(overlap-1);

	if(featureType == "rects")
	{
		rectFiltNum = calcRectFiltNum(charSizeX,charSizeY)+1;
		featureMat = Mat::zeros(1,rectFiltNum,CV_32FC1);
	}
	else if(featureType == "points")
	{
		featureMat = Mat::zeros(1,numOfPointPairs,CV_32FC1);
	}

	Mat treePred;
	CvForestTree* tree;
	double minVal, maxVal;
	int minIndx[2] = {0,0};
	int maxIndx[2] = {0,0};
	CvMat sample1, sample2;
	float proxOld, proxNew;
	char prox;
	int numOfForests = (int)forestVector.size();
	Mat pointPairVector = Mat::zeros(numOfPointPairs,4,CV_32SC1);
	int charSizeXUpSampled = charSizeX*downSample;
	int charSizeYUpSampled = charSizeY*downSample;

	if(featureType == "points")
	{
		cv::RNG rng(0);
		int distThreshold = 20;
		int x1, x2, y1, y2;
		for(int i=0; i<numOfPointPairs; i++)
		{
			x1 = 0;
			y1 = 0;
			x2 = 0;
			y2 = 0;
			
			while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
			{
				x1 = rng.uniform(charSizeXUpSampled/4,charSizeXUpSampled*3/4);
				y1 = rng.uniform(charSizeYUpSampled/4,charSizeYUpSampled*3/4);
				x2 = rng.uniform(0,charSizeXUpSampled);
				y2 = rng.uniform(0,charSizeYUpSampled);
			}
			pointPairVector.at<int>(i,0) = x1/downSample;
			pointPairVector.at<int>(i,1) = y1/downSample;
			pointPairVector.at<int>(i,2) = x2/downSample;
			pointPairVector.at<int>(i,3) = y2/downSample;
		}
	}

	start = GetTickCount();
	for(int im=0; im<imNum; im++)
	{
		pred = new Mat(Mat::zeros(tileNumY,tileNumX, CV_8UC1));
		xPos = 0;
		yPos = 0;
		predPosx = 0;
		predPosy = 0;

		while(yPos < imageHeight - charSizeY)
		{
			while(xPos < imageWidth - charSizeX)
			{
				int rectArea = charSizeX*charSizeY*255;
				imRect = (*imageVector[im])(Rect(xPos,yPos,charSizeX,charSizeY));
				imRectSum = sum(imRect)(0);
				cv::threshold(imRect,imRectThres,60,255,CV_8UC1);
				//cv::threshold(imRect,imRect,60,255,CV_8UC1);
				imRectUp = sum(imRectThres(Rect(0,0,charSizeX,charSizeY/4)))(0);
				imRectDown = sum(imRectThres(Rect(0,charSizeY*3/4,charSizeX,charSizeY/4)))(0);
				imRectMiddleHor = sum(imRectThres(Rect(0,charSizeY*7/16,charSizeX,charSizeY/8)))(0);
				imRectMiddleVert = sum(imRectThres(Rect(charSizeX*7/16,0,charSizeX/8, charSizeY)))(0);

				if(imRectSum > rectArea/2 && imRectUp < rectArea/4 && imRectDown < rectArea/4 && imRectMiddleVert < rectArea/8 && imRectMiddleHor < rectArea/8)
				{
					//imshow("sfsdf",imRect);
					//waitKey();
					if(featureType == "rects")
						calcRectFeatureTile(imRect,featureMat,charSizeX,charSizeY,0);
					else if(featureType == "points")
					{
						featureMat = Mat::zeros(1,numOfPointPairs,CV_32FC1);
						calcPointPairsFeaturesTile(imRect,featureMat,pointPairVector,numOfPointPairs,0, useNoise);
					}
					else
						abort();

					//Loop over all trees

					treePred = Mat::zeros(256,1, CV_32SC1);

					for(int f=0; f<numOfForests; f++)
					{
						for(int t=0; t<(int)numOfTrees; t++)
						{
							tree = forestVector[f]->get_tree(t);
							treePred.at<int>(tree->predict(featureMat)->value,0)++;
						}
					}
					cv::minMaxIdx(treePred,&minVal,&maxVal,minIndx,maxIndx);
					//cout << (char)(*maxIndx) << "\t" << maxVal/(numOfForests*numOfTrees) << endl;
					if(maxVal > numOfTrees*numOfForests*desicionThres)
						pred->at<uchar>(predPosy,predPosx) = *maxIndx;
				}
				predPosx++;
				xPos += overlapTileX;
			}
			xPos = 0;
			yPos += overlapTileY;
			predPosx = 0;
			predPosy++;
		}
		predictions.push_back(pred);
	}
	stop = GetTickCount();
	std::cout << "Average time per image: " << (float)(stop - start)/((float)imNum)/1000 << std::endl << std::endl; 
	return predictions;
}

void evaluateResultRealImage(vector<Mat*> predictions, vector<Mat*> imageVector,  int imageWidth, int imageHeight, int charSizeX, int charSizeY, int numOfImages, int overlap, float downSample)
{
	printf("Visulize result....\n\n");
	int overlapTileX = charSizeX/overlap;
	int overlapTileY = charSizeY/overlap;
	int tileNumX = imageWidth/charSizeX*overlap - (overlap-1);
	int tileNumY = imageHeight/charSizeY*overlap - (overlap-1);
	char p;
	int xPos, yPos, predPosx, predPosy;
	int numOfTrue, numOfFalse;
	char maxRes;
	string charStr;
	double charSizeR = 0.25;//charSize/(100*overlap) + 1;
	Mat characterRect;
	Mat visulizePred = Mat::zeros(imageHeight,imageWidth,CV_8UC3);
	add(visulizePred,255,visulizePred);

	for(int i=0; i<numOfImages; i++)
	{
		numOfTrue = 0;
		numOfFalse = 0;
		xPos = 0;
		yPos = 0;
		predPosx = 0;
		predPosy = 0;

		while(yPos < imageHeight - charSizeY)
		{
			while(xPos < imageWidth - charSizeX)
			{
				//cout << x << endl;

				if(predictions[i]->at<uchar>(predPosy,predPosx))
				{
					p = predictions[i]->at<uchar>(predPosy,predPosx);
					charStr = p;
					characterRect = Mat::zeros(overlapTileY,overlapTileX,CV_8UC3);
					add(characterRect,255,characterRect);
					cv::Point org(overlapTileX/12,overlapTileY-overlapTileY/12);
					cv::putText(characterRect,charStr , org, 0, charSizeR,CV_RGB(0,0,255), 1, 8,false);
					numOfTrue++;
					characterRect.copyTo(visulizePred(Rect(xPos,yPos,overlapTileX,overlapTileY)));
				}

				predPosx++;
				xPos += overlapTileX;
			}
			xPos = 0;
			yPos += overlapTileY;
			predPosx = 0;
			predPosy++;
		}
		cout << "Number of true detections: " << numOfTrue << endl;
		cout << "Number of false detections: " << numOfFalse << endl;
		imshow("image", *imageVector[i]);
		imshow("visulize predictions",visulizePred);
		waitKey();
	}
}
