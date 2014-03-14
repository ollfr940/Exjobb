#include "stdafx.h"
#include "functions.h"
using namespace std;
using namespace cv;

vector<Mat*> produceTrainingData(int num, int imageSize)
{
	vector<Mat*> imageVec;
	Mat* image;
	cv::RNG rng;
	char d = '0';
	string dstr;

	for(int c=0; c<10; c++)
	{
	for(int j=0; j<num; j++)
	{
		image = new Mat(Mat::zeros(imageSize,imageSize,CV_8UC1));
		cv::add(*image,255,*image);
		cv::Point org;
		org.x = 10;
		org.y = imageSize-10;
		dstr = d;
		cv::putText(*image,dstr , org, 0, 5 ,0, 4, 8);
		d++;

		imageVec.push_back(image);
		//cv::imshow("im", image);
		//cv::waitKey();

	}
	}
	return imageVec;
}


Mat createRectFeatures(vector<Mat*> trainingData, int trainingNum, int imageSize)
{
	Mat integralIm, integralRect;
	int p;
	Mat featureMat = Mat::zeros(trainingData.size(),8*8*17,CV_8UC1);
	for(int im=0; im<trainingData.size(); im++)
	{
		//integral(*trainingData[im], integralIm);
		for(int rectx=0; rectx <8; rectx++)
		{
			for(int recty=0; recty <8; recty++)
			{
				int rectSizex = 12 + rectx*4;
				int rectSizey = 12 + recty*4;
				int rectNumx = imageSize/rectSizex;
				int rectNumy = imageSize/rectSizey;

				for(int i=0; i<rectNumx; i++)
				{
					for(int j=0; j<rectNumy; j++)
					{
						integral((*trainingData[im])(Rect(i*rectSizex,j*rectSizey,rectSizex,rectSizey)),integralRect);

						p = integralRect.at<uchar>(rectSizex-1,rectSizey-1) - integralRect.at<uchar>(rectSizex*0.5,rectSizey-1);
						featureMat.at<uchar>(im, rectx*17+recty*17) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex-1,rectSizey-1) - integralRect.at<uchar>(rectSizex-1,rectSizey/2);
						featureMat.at<uchar>(im, rectx*17+recty*17+1) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex*0.75,rectSizey-1) - integralRect.at<uchar>(rectSizex*0.25,rectSizey-1);
						featureMat.at<uchar>(im, rectx*17+recty*17+2) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex-1,rectSizey*0.75) - integralRect.at<uchar>(rectSizex-1,rectSizey*0.25);
						featureMat.at<uchar>(im, rectx*17+recty*17+3) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex*0.75,rectSizey*0.75) - integralRect.at<uchar>(rectSizex*0.25,rectSizey*0.75) - integralRect.at<uchar>(rectSizex*0.75,rectSizey*0.25) + integralRect.at<uchar>(rectSizex*0.25,rectSizey*0.25);
						featureMat.at<uchar>(im, rectx*17+recty*17+4) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex-1,rectSizey-1) -  integralRect.at<uchar>(rectSizex*0.25,rectSizey-1) - integralRect.at<uchar>(rectSizex-1,rectSizey*0.25) + integralRect.at<uchar>(rectSizex*0.25,rectSizey*0.25);
						featureMat.at<uchar>(im, rectx*17+recty*17+5) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex*0.75,rectSizey-1) - integralRect.at<uchar>(rectSizex*0.75,rectSizey*0.25);
						featureMat.at<uchar>(im, rectx*17+recty*17+6) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex*0.75,rectSizey*0.75);
						featureMat.at<uchar>(im, rectx*17+recty*17+7) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex-1,rectSizey*0.75) - integralRect.at<uchar>(rectSizex*0.25,rectSizey*0.75);
						featureMat.at<uchar>(im, rectx*17+recty*17+8) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex-1,rectSizey*0.75) - integralRect.at<uchar>(rectSizex*0.25,rectSizey*0.75) - integralRect.at<uchar>(rectSizex-1,rectSizey*0.25) + integralRect.at<uchar>(rectSizex-1*0.25,rectSizey-1*0.25);
						featureMat.at<uchar>(im, rectx*17+recty*17+9) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex*0.75,rectSizey*0.75) - integralRect.at<uchar>(rectSizex*0.75,rectSizey*0.25);
						featureMat.at<uchar>(im, rectx*17+recty*17+10) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex*0.75,rectSizey*0.75) - integralRect.at<uchar>(rectSizex*0.25,rectSizey*0.75);
						featureMat.at<uchar>(im, rectx*17+recty*17+11) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex*0.75,rectSizey-1) - integralRect.at<uchar>(rectSizex*0.25,rectSizey-1) - integralRect.at<uchar>(rectSizex*0.25,rectSizey*0.75) + integralRect.at<uchar>(rectSizex*0.25,rectSizey*0.25);
						featureMat.at<uchar>(im, rectx*17+recty*17+12) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex-1,rectSizey-1) - integralRect.at<uchar>(rectSizex*0.75,rectSizey-1);
						featureMat.at<uchar>(im, rectx*17+recty*17+13) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex*0.25,rectSizey-1);
						featureMat.at<uchar>(im, rectx*17+recty*17+14) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex-1,rectSizey-1) - integralRect.at<uchar>(rectSizex-1,rectSizey*0.75);
						featureMat.at<uchar>(im, rectx*17+recty*17+15) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
						p = integralRect.at<uchar>(rectSizex-1,rectSizey*0.25);
						featureMat.at<uchar>(im, rectx*17+recty*17+16) += 2*p - integralRect.at<uchar>(rectSizex-1,rectSizey-1);
					}
				}
			}
		}
	}
	return featureMat;
}