#include"stdafx.h"
#include"features.h"

using namespace std;
using namespace cv;


void calcRectFeatureTile(Mat& tile, Mat& featureMat, int width, int height, int im)
{
	Mat integralIm, integralRect;
	float p;
	int rectFiltNum = calcRectFiltNum(width,height)+1;
	int indx = 0; 
	for(int rectx=0; rectx <8; rectx++)
	{
		for(int recty=0; recty <8; recty++)
		{
			int rectSizex = 12 + rectx*4;
			int rectSizey = 12 + recty*4;
			int rectNumx = width/rectSizex;
			int rectNumy = height/rectSizey;

			for(int i=0; i<rectNumx; i++)
			{
				for(int j=0; j<rectNumy; j++)
				{
					integral(tile(Rect(i*rectSizex,j*rectSizey,rectSizex,rectSizey)),integralRect,CV_32FC1);
					p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey/2,rectSizex);
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
			}
		}
	}
}

void calcPointPairsFeaturesTile(Mat& tile, Mat& featureMat, Mat& pointVector, int numOfPoints, int im, bool useNoise)
{
	int x1, y1, x2, y2;
	//int n=0;
	for(int i=0; i<numOfPoints; i++)
	{
		x1 = pointVector.at<int>(i,0);
		y1 = pointVector.at<int>(i,1);
		x2 = pointVector.at<int>(i,2);
		y2 = pointVector.at<int>(i,3);

		if(useNoise)
		{
			if(tile.at<uchar>(y1,x1) > tile.at<uchar>(y2,x2))
				featureMat.at<float>(im, i) = 1;
			else if(tile.at<uchar>(y1,x1) < tile.at<uchar>(y2,x2))
				featureMat.at<float>(im,i) = -1;
		}
		else
		{
			if(tile.at<uchar>(y1,x1) > tile.at<uchar>(y2,x2) + 15)
				featureMat.at<float>(im, i) = 1;
			else if(tile.at<uchar>(y1,x1) < tile.at<uchar>(y2,x2)-15)
				featureMat.at<float>(im,i) = -1;
		}
		//if(featureMat.at<float>(im,i) != 0)
			//n++;
	}
	//if(n > 1000)
		//cout <<  " sdf " << n << endl;
}

void calcLinesFeaturesTile(Mat& tile, Mat& featureMat, Mat& pointVector, int numOfLines, int im)
{
	int x, y, x1, y1, x2, y2;
	float k, m;	
	bool v = true;
	for(int i=0; i<numOfLines; i++)
	{
		x1 = pointVector.at<int>(i,0);
		y1 = pointVector.at<int>(i,1);
		x2 = pointVector.at<int>(i,2);
		y2 = pointVector.at<int>(i,3);
		k = (float)(y2-y1)/(float)(x2-x1);
		m = y1 - k*x1;

		if(x1 < x2)
			x = x1;
		else
		{
			x = x2;
			x2 = x1;
		}
		for(; x<x2; x++)
		{
			y = static_cast<int>(k*x + m);
			if(tile.at<uchar>(x,y) == 0 && v)
			{
				featureMat.at<float>(im,i)++;
				v = false;
			}
			else
				v = true;
		}		
	}
}

void calcStdTile(Mat& tile, Mat& featureMat, int im, int reSizeTo)
{
	//Scalar mean, std, meanTile, stdTile;
	Mat tileRect;
	int rectSizeX = tile.cols/reSizeTo;
	int rectSizeY = tile.rows/reSizeTo;
	//cv::meanStdDev(tile,mean,std);

	for(int y=0; y<reSizeTo; y++)
	{
		for(int x=0; x<reSizeTo; x++)
		{
			tileRect = tile(Rect(x*rectSizeX,y*rectSizeY,rectSizeX,rectSizeY));
			//cv::meanStdDev(tileRect,meanTile,stdTile);
			//cout << stdTile(0) << endl;
			featureMat.at<float>(im,y*reSizeTo+x) = sum(tileRect)(0); //(meanTile(0) - mean(0))/std(0);

		}
	}
}

Mat calcFeaturesTraining(RandomCharacters trainingData, int numOfPoints, string featureType, int tileSizeX, int tileSizeY, bool useNoise)
{
	int numOfCharacters = static_cast<int>(trainingData.randChars.size());

	if(featureType == "rects")
	{
		printf("Create rect features....\n\n");
		int rectFiltNum = calcRectFiltNum(tileSizeX,tileSizeY)+1;
		Mat featureMat = Mat::zeros(numOfCharacters,rectFiltNum,CV_32FC1);

		for(int im=0; im<numOfCharacters; im++)
		{
			calcRectFeatureTile(*trainingData.randChars[im], featureMat, tileSizeX, tileSizeY, im);
		}

		return featureMat;
	}

	else if(featureType == "points")
	{
		printf("Create random point pairs features....\n\n");
		Mat pointPairVector = Mat::zeros(numOfPoints,4,CV_32SC1);
		Mat featureMat = Mat::zeros(numOfCharacters,numOfPoints,CV_32FC1);
		cv::RNG rng(0);
		int distThreshold = 10;
		int x1, x2, y1, y2;
		for(int i=0; i<numOfPoints; i++)
		{
			x1 = 0;
			y1 = 0;
			x2 = 0;
			y2 = 0;

			while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
			{
				x1 = rng.uniform(tileSizeX/4,tileSizeX*3/4);
				y1 = rng.uniform(tileSizeY/4,tileSizeY*3/4);
				x2 = rng.uniform(0,tileSizeX);
				y2 = rng.uniform(0,tileSizeY);
			}

			pointPairVector.at<int>(i,0) = x1;
			pointPairVector.at<int>(i,1) = y1;
			pointPairVector.at<int>(i,2) = x2;
			pointPairVector.at<int>(i,3) = y2;
		}

		for(int im=0; im<numOfCharacters; im++)
		{ 
			calcPointPairsFeaturesTile(*trainingData.randChars[im],featureMat,pointPairVector, numOfPoints, im, useNoise);
		}

		return featureMat;
	}
	else if(featureType == "Lines")
	{
		printf("Create random Lines features....\n\n");
		Mat LineVector = Mat::zeros(numOfPoints,4,CV_32FC1);
		Mat featureMat = Mat::zeros(numOfCharacters,numOfPoints,CV_32FC1);
		cv::RNG rng(0);
		int distThreshold = 60;
		int x1, x2, y1, y2;
		for(int i=0; i<numOfPoints; i++)
		{
			x1 = 0;
			y1 = 0;
			x2 = 0;
			y2 = 0;

			while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
			{
				x1 = rng.uniform(tileSizeX/4,tileSizeX*3/4);
				y1 = rng.uniform(tileSizeY/4,tileSizeY*3/4);
				x2 = rng.uniform(0,tileSizeX);
				y2 = rng.uniform(0,tileSizeY);
			}

			LineVector.at<int>(i,0) = x1;
			LineVector.at<int>(i,1) = y1;
			LineVector.at<int>(i,2) = x2;
			LineVector.at<int>(i,3) = y2;
		}

		for(int im=0; im<numOfCharacters; im++)
		{ 
			//calcPointPairsFeaturesTile(*trainingData.randChars[im],featureMat,LineVector, numOfPoints, im, downSample);
		}

		return featureMat;
	}
	else
		abort();
}

RandomImagesAndCharacters calcPointPairFeaturesScales(vector<Mat*> randomImages, RandomCharacters trainingData, int tileSizeX, int tileSizeY, int resizeTo, int imageSize, int numOfPointPairs, 
	int numOfTrueCharacters, int numOfFalseCharacters, string typeOfChars, bool useNoise)
{
	printf("Calculate random point pairs features for detection of false tiles....\n\n");
	RandomImagesAndCharacters r;
	int numOfFalseImages = randomImages.size()*(imageSize/tileSizeX)*(imageSize/tileSizeY);
	
	if(typeOfChars == "digitsAndLetters")
		numOfTrueCharacters = numOfTrueCharacters*36;
	else
		abort();

	int tileNumX = imageSize/tileSizeX;
	int tileNumY = imageSize/tileSizeY;
	r.features = Mat::zeros(numOfTrueCharacters+numOfFalseCharacters+numOfFalseImages,numOfPointPairs, CV_32FC1);
	r.responses = Mat::zeros(numOfTrueCharacters+numOfFalseCharacters+numOfFalseImages,1,CV_32SC1);
	Mat imRect;
	Mat pointPairVector = Mat::zeros(numOfPointPairs,4,CV_32SC1);
	cv::RNG rng(0);
	int distThreshold = 10;
	int x1, x2, y1, y2;
	for(int i=0; i<numOfPointPairs; i++)
	{
		x1 = 0;
		y1 = 0;
		x2 = 0;
		y2 = 0;

		while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
		{
			x1 = rng.uniform(tileSizeX/4,tileSizeX*3/4);
			y1 = rng.uniform(tileSizeY/4,tileSizeY*3/4);
			x2 = rng.uniform(0,tileSizeX);
			y2 = rng.uniform(0,tileSizeY);
		}

		pointPairVector.at<int>(i,0) = x1/(tileSizeX/resizeTo);
		pointPairVector.at<int>(i,1) = y1/(tileSizeY/resizeTo);
		pointPairVector.at<int>(i,2) = x2/(tileSizeX/resizeTo);
		pointPairVector.at<int>(i,3) = y2/(tileSizeY/resizeTo);
	}


	for(int i=0; i< randomImages.size(); i++)
	{
		for(int y=0; y<tileNumY; y++)
		{
			for(int x=0; x<tileNumX; x++)
			{
				imRect = (*randomImages[i])(Rect(x,y,tileSizeX,tileSizeY));
				resize(imRect,imRect,Size(resizeTo,resizeTo));
				calcPointPairsFeaturesTile(imRect, r.features, pointPairVector,numOfPointPairs,i*tileNumX*tileNumY+y*tileNumY+x,useNoise);
			}
		}
		delete randomImages[i];
	}

	for(int i=0; i<numOfTrueCharacters; i++)
	{
		resize(*trainingData.randChars[i],imRect,Size(resizeTo,resizeTo));
		calcPointPairsFeaturesTile(imRect,r.features,pointPairVector,numOfPointPairs,numOfFalseImages+i,useNoise);
		r.responses.at<int>(numOfFalseImages+i,0) = 1;
		delete trainingData.randChars[i];
	}

	for(int i=numOfTrueCharacters; i<numOfTrueCharacters+numOfFalseCharacters; i++)
	{
		resize(*trainingData.randChars[i],imRect,Size(resizeTo,resizeTo));
		calcPointPairsFeaturesTile(imRect,r.features,pointPairVector,numOfPointPairs,numOfFalseImages+i,useNoise);
		delete trainingData.randChars[i];
	}
	return r;
}

RandomImagesAndCharacters calcStandardDeviationFeatures(vector<Mat*> randomImages, RandomCharacters trainingData, int tileSizeX, int tileSizeY, int imageSize, 
	int numOfTrueCharacters, int numOfFalseCharacters, string typeOfChars, int reSizeTo)
{
	int numOfFeatures = reSizeTo*reSizeTo;
	printf("Calculate standard deviation features for detection of false tiles....\n\n");
	RandomImagesAndCharacters r;
	int numOfFalseImages = randomImages.size()*(imageSize/tileSizeX)*(imageSize/tileSizeY);
	
	if(typeOfChars == "digitsAndLetters")
		numOfTrueCharacters = numOfTrueCharacters*36;
	else
		abort();

	int tileNumX = imageSize/tileSizeX;
	int tileNumY = imageSize/tileSizeY;
	r.features = Mat::zeros(numOfTrueCharacters+numOfFalseCharacters+numOfFalseImages,numOfFeatures, CV_32FC1);
	r.responses = Mat::zeros(numOfTrueCharacters+numOfFalseCharacters+numOfFalseImages,1,CV_32SC1);
	Mat imRect;

	for(int i=0; i< randomImages.size(); i++)
	{
		for(int y=0; y<tileNumY; y++)
		{
			for(int x=0; x<tileNumX; x++)
			{
				imRect = (*randomImages[i])(Rect(x,y,tileSizeX,tileSizeY));
				calcStdTile(imRect,r.features, i*tileNumX*tileNumY+y*tileNumY+x, reSizeTo);
			}
		}
		delete randomImages[i];
	}

	for(int i=0; i<numOfTrueCharacters; i++)
	{
		calcStdTile(*trainingData.randChars[i],r.features,numOfFalseImages+i, reSizeTo);
		r.responses.at<int>(numOfFalseImages+i,0) = 1;
		delete trainingData.randChars[i];
	}

	for(int i=numOfTrueCharacters; i<numOfTrueCharacters+numOfFalseCharacters; i++)
	{
		calcStdTile(*trainingData.randChars[i],r.features,numOfFalseImages+i, reSizeTo);
		delete trainingData.randChars[i];
	}
	return r;
}
/*Mat calcPointPairsFeaturesTraining(RandomCharacters trainingData, int numOfPoints)
{
printf("Create random point pairs features....\n\n");

printf("Create random point pairs features....\n\n");

Mat pointPairVector = Mat::zeros(numOfPoints,4,CV_32FC1);
int width = trainingData.randChars[0]->size().width;
int height = trainingData.randChars[0]->size().height;
int numOfCharacters = (int)trainingData.randChars.size();
Mat featureMat = Mat::zeros(numOfCharacters,numOfPoints,CV_32FC1);
//uint64 initValue = time(0);
cv::RNG rng(0);
for(int i=0; i<numOfPoints; i++)
{
pointPairVector.at<int>(i,0) = rng.uniform(0,width);
pointPairVector.at<int>(i,1) = rng.uniform(0,height);
pointPairVector.at<int>(i,2) = rng.uniform(0,width);
pointPairVector.at<int>(i,3) = rng.uniform(0,height);
}

for(int im=0; im<numOfCharacters; im++)
{ 
calcPointPairsFeaturesTile(*trainingData.randChars[im],featureMat,pointPairVector, numOfPoints, im);
}
return featureMat;
}*/