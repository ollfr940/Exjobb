#include "stdafx.h"
#include <iostream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef CLASSES_H
#define CLASSES_H



struct CSize
{
	int width;
	int height;
};

struct RandomCharacters
{
	std::vector<cv::Mat*> randChars;
	cv::Mat responses;
};

struct RandomImagesAndCharacters
{
	cv::Mat features;
	cv::Mat responses;
};

struct RandomCharactersImages
{
	std::vector<cv::Mat*> randChars;
	std::vector<cv::Mat*> responses;
};
/*
class CalcSample
{
public:
	CalcSample() {}
	virtual void operator() (cv::Mat& tile, cv::Mat& featureMat, int& indx, int rectSizex, int rectSizey, int im) = 0;
};


class CalcRectSample : public CalcSample
{
public:
	CalcRectSample() : CalcSample() {}

	void operator() (cv::Mat& integralRect, cv::Mat& featureMat, int& indx, int rectSizex, int rectSizey, int im)
	{
		float p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey/2,rectSizex);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey,rectSizex/2);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey*3/4,rectSizex) - integralRect.at<float>(rectSizey/4,rectSizex);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey,rectSizex*3/4) - integralRect.at<float>(rectSizey,rectSizex/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey*3/4,rectSizex*3/4) - integralRect.at<float>(rectSizey/4,rectSizex*3/4) - integralRect.at<float>(rectSizey*3/4,rectSizex/4) + integralRect.at<float>(rectSizey/4,rectSizex/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey,rectSizex) -  integralRect.at<float>(rectSizey/4,rectSizex) - integralRect.at<float>(rectSizey,rectSizex/4) + integralRect.at<float>(rectSizey/4,rectSizex/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey*3/4,rectSizex) - integralRect.at<float>(rectSizey*3/4,rectSizex/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey*3/4,rectSizex*3/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey,rectSizex*3/4) - integralRect.at<float>(rectSizey/4,rectSizex*3/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey,rectSizex*3/4) - integralRect.at<float>(rectSizey/4,rectSizex*3/4) - integralRect.at<float>(rectSizey,rectSizex/4) + integralRect.at<float>(rectSizey/4,rectSizex/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey*3/4,rectSizex*3/4) - integralRect.at<float>(rectSizey*3/4,rectSizex/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey*3/4,rectSizex*3/4) - integralRect.at<float>(rectSizey/4,rectSizex*3/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey*3/4,rectSizex) - integralRect.at<float>(rectSizey/4,rectSizex) - integralRect.at<float>(rectSizey/4,rectSizex*3/4) + integralRect.at<float>(rectSizey/4,rectSizex/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey-1,rectSizex-1);
		indx++;
		p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey*3/4,rectSizex);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey/4,rectSizex);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey,rectSizex*3/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
		p = integralRect.at<float>(rectSizey,rectSizex/4);
		featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
		indx++;
	}

};*/
/*
class CalcRandomPointPairsSample
{
public:
	CalcRandomPointPairsSample() {}

	void operator() (cv::Mat& pointVector, cv::Mat& featureMat, cv::Mat* image, int numOfPoints, int im)
	{
		int x1, y1, x2, y2;
		for(int i=0; i<numOfPoints; i++)
		{
			x1 = pointVector.at<int>(i,0);
			y1 = pointVector.at<int>(i,1);
			x2 = pointVector.at<int>(i,2);
			y2 = pointVector.at<int>(i,3);
			if(image->at<float>(x1,y1) > image->at<float>(x2,y2))
				featureMat.at<float>(im, i) = 1;
		}
	}
};
*/

#endif