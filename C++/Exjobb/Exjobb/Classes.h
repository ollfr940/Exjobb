#include "stdafx.h"
#include <iostream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>

#ifndef CLASSES_H
#define CLASSES_H

class CalcSample
{
public:
	CalcSample(int f,cv::Mat t, bool s) : features(f), testSample(t), testing(s) {}
	virtual cv::Mat operator() (cv::Mat& tile) = 0;
	int features;
	cv::Mat testSample;
	std::string name;
	cv::Mat sobxTile, sobyTile, distanceTile;
	bool testing;
};


class CalcSTDSample : public CalcSample
{
public:
	CalcSTDSample(int f,cv::Mat t, bool s) : CalcSample(f,t,s)
	{
		name = "STD";
	}

	cv::Mat operator() (cv::Mat& tile)
	{
		cv::meanStdDev(tile,mean,std);
		
		if(testing)
		{
			testSample.at<float>(0,0) = 0;
			testSample.at<float>(0,1) = (float)std.at<double>(0,0);
		}
		else
			testSample.at<float>(0,0) = (float)std.at<double>(0,0);

		return testSample;
	}

	cv::Mat mean, std;
};

class CalcFASTSample : public CalcSample
{
public:
	CalcFASTSample(int f,cv::Mat t, bool s) : CalcSample(f,t,s) 
	{
		name = "FAST";
	}
	cv::Mat operator() (cv::Mat& tile)
	{
		if(testing)
			testSample.at<float>(0,0) = 0;

		cv::FAST(tile,keyPoint,10,true);
		if(testing)
			testSample.at<float>(0,1) = (float)keyPoint.size();
		else
			testSample.at<float>(0,0) = (float)keyPoint.size();

		cv::FAST(tile,keyPoint,25,true);
		if(testing)
			testSample.at<float>(0,2) = (float)keyPoint.size();
		else
			testSample.at<float>(0,1) = (float)keyPoint.size();

		cv::FAST(tile,keyPoint,50,true);
		if(testing)
			testSample.at<float>(0,3) = (float)keyPoint.size();
		else
			testSample.at<float>(0,2) = (float)keyPoint.size();

		return testSample;
	}
	std::vector<cv::KeyPoint> keyPoint;
};

class CalcLBPSample : public CalcSample
{
public:
	CalcLBPSample(int f,cv::Mat t, bool s, cv::Mat bin, int ts) : CalcSample(f,t,s)
	{
		name = "LBP";
		binary = bin;
		tileSize = ts;
		channels[0] = 0; channels[1] = 1; 
		histSize[0] = 256;
		range[0] = 0; range[1] = 256;
	}
	cv::Mat operator() (cv::Mat& tile)
	{
		//testSample.at<float>(0,0) = 'T';
		const float* ranges[] = {range};
		for(int ii=0; ii<tileSize-2; ii++)
		{
			for(int jj=0; jj<tileSize-2; jj++)
			{
				block = tile(cv::Rect(ii,jj,3,3));
				cv::compare(block,block.at<uchar>(1,1),LBPblock,cv::CMP_GT);
				LBPblock.convertTo(bp,CV_32FC1);
				binary.at<float>(ii,jj) = (bp.at<float>(0,0)+bp.at<float>(1,0)*2+bp.at<float>(2,0)*4+bp.at<float>(2,1)*8+
					bp.at<float>(2,2)*16+bp.at<float>(1,2)*32+bp.at<float>(0,2)*64+bp.at<float>(0,1)*128)/255;
			}
		}
		calcHist(&binary,1,channels,cv::Mat(),hist,1,histSize,ranges,true,false);

		if(testing)
		{
			testSample.at<float>(0,0) = 0;
			for(int f=0; f<features; f++)
				testSample.at<float>(0,f+1) = hist.at<float>(f);
		}
		else
		{
			for(int f=0; f<features; f++)
				testSample.at<float>(0,f) = hist.at<float>(f);
		}

		return testSample;
	}

	cv::Mat binary;
	cv::Mat LBPblock, block, bp, mean, std;
	cv::MatND hist;
	int tileSize;
	int channels[2];
	int	histSize[1];
	float range[2];
};

class CalcI1DSample : public CalcSample
{
public:
	CalcI1DSample(int f,cv::Mat t, bool s) : CalcSample(f,t,s)
	{
		name = "I1D";
		structureTensor = cv::Mat::zeros(2,2,CV_32FC1);
	}
	cv::Mat operator() (cv::Mat& tile)
	{
		if(testing)
		{
			cv::GaussianBlur( tile, gaussianTile, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
			cv::Sobel( gaussianTile, sobxTile, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );
			cv::Sobel(gaussianTile, sobyTile, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT );
		}

		structureTensor.at<float>(0,0) = (float)sobxTile.dot(sobxTile);
		structureTensor.at<float>(0,1) = (float)sobxTile.dot(sobyTile);
		structureTensor.at<float>(1,0) = (float)sobxTile.dot(sobyTile);
		structureTensor.at<float>(1,1) = (float)sobyTile.dot(sobyTile);
		cv::eigen(structureTensor, eigenvalues);
		score1D = cv::pow((eigenvalues[0]-eigenvalues[1]),2)/(pow(eigenvalues[0],2)+pow(eigenvalues[1],2));

		if(testing)
		{
			testSample.at<float>(0,0) = 0;
			testSample.at<float>(0,1) = score1D;
			//testSample.at<float>(0,1) = eigenvectors.at<float>(0,0);
			//testSample.at<float>(0,2) = eigenvectors.at<float>(0,1);
		}
		else
		{
			testSample.at<float>(0,0) = score1D;
			//testSample.at<float>(0,0) = eigenvectors.at<float>(0,0);
			//testSample.at<float>(0,1) = eigenvectors.at<float>(0,1);
		}

		return testSample;
	}

	cv::Mat structureTensor, gaussianTile;
	float score1D;
	std::vector<float> eigenvalues;
	//cv::Mat eigenvectors;
};

class CalcDistSample : public CalcSample
{
public:
	CalcDistSample(int f,cv::Mat t, bool s) : CalcSample(f,t,s)
	{
		name = "Distance";
		//threshold1 = t1;
		//threshold2 = t2;
	}
	cv::Mat operator() (cv::Mat& tile)
	{
		if(testing)
		{
			cv::Canny(tile,canny,100,300);
			cv::threshold(canny,canny,128,255, cv::THRESH_BINARY_INV);
			cv::distanceTransform(canny,distanceTile,CV_DIST_L1,3);
		}
		cv::meanStdDev(distanceTile, mean, std);

		if(testing)
		{
			testSample.at<float>(0,0) = 0;
			testSample.at<float>(0,1) = (float)mean.at<double>(0,0);
			testSample.at<float>(0,2) = (float)std.at<double>(0,0);
		}
		else
		{
			testSample.at<float>(0,0) = (float)mean.at<double>(0,0);
			testSample.at<float>(0,1) = (float)std.at<double>(0,0);
		}

		return testSample;
	}

	cv::Mat mean, std, canny;
	//int threshold1, threshold2;
};
#endif