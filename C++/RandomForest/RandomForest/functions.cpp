#include "stdafx.h"
#include "functions.h"
using namespace std;
using namespace cv;

void writeMatToFile(Mat& m, Mat& r,int imageNum, const char* filename)
{
	std::ofstream fout(filename);

	if(!fout)
	{
		std::cout<<"File Not Opened"<<std::endl;  return;
	}

	for(int i=0; i<m.rows; i++)
	{
		fout << r.at<int>(i,0) << ',';
		for(int j=0; j<m.cols-1; j++)
		{
			fout<<m.at<float>(i,j)<<',';
		}
		fout << m.at<float>(i,m.cols-1);
		fout<<std::endl;
	}

	fout.close();
}

vector<Mat*> produceData(int first, int characters, int num, int imageSize)
{
	vector<Mat*> imageVec;
	Mat* image;
	int charSize = (int)imageSize/30 + 1;
	cv::RNG rng;
	char d = '0';
	d += first;
	string dstr;

	for(int c=0; c<characters; c++)
	{
		for(int j=0; j<num; j++)
		{
			image = new Mat(Mat::zeros(120,120,CV_8UC1));
			cv::add(*image,255,*image);
			cv::Point org;
			org.x = 10;
			org.y = 10;
			dstr = d;
			cv::putText(*image,dstr , org, 0, charSize ,0, 4, 8,true);

			imageVec.push_back(image);
			cv::imshow("im", *image);
			cv::waitKey();

		}
		d++;
	}
	return imageVec;
}

Mat createResponses(int trainingNum, int characters)
{
	int res = 0;
	Mat responses = Mat::zeros(trainingNum*characters,1,CV_32SC1);
	for(int i = 0; i< characters; i++)
	{
		for(int j=0; j<trainingNum; j++)
			responses.at<int>(i*trainingNum+j,0) = res;
		res++;
	}
	return responses;
}

Mat createRectFeatures(vector<Mat*> trainingData, int trainingNum, int imageSize)
{
	Mat integralIm, integralRect;
	float p;
	int filtNum = 28578;
	Mat featureMat = Mat::zeros(trainingData.size(),filtNum,CV_32FC1);
	for(int im=0; im<trainingData.size(); im++)
	{
		//integral(*trainingData[im], integralIm);
		int indx = 0; 
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
						integral((*trainingData[im])(Rect(i*rectSizex,j*rectSizey,rectSizex,rectSizey)),integralRect,CV_32FC1);
		
						p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey*0.5,rectSizex);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey,rectSizex/2);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey*0.75,rectSizex) - integralRect.at<float>(rectSizey*0.25,rectSizex);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey,rectSizex*0.75) - integralRect.at<float>(rectSizey,rectSizex*0.25);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey*0.75,rectSizex*0.75) - integralRect.at<float>(rectSizey*0.25,rectSizex*0.75) - integralRect.at<float>(rectSizey*0.75,rectSizex*0.25) + integralRect.at<float>(rectSizey*0.25,rectSizex*0.25);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey,rectSizex) -  integralRect.at<float>(rectSizey*0.25,rectSizex) - integralRect.at<float>(rectSizey,rectSizex*0.25) + integralRect.at<float>(rectSizey*0.25,rectSizex*0.25);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey*0.75,rectSizex) - integralRect.at<float>(rectSizey*0.75,rectSizex*0.25);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey*0.75,rectSizex*0.75);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey,rectSizex*0.75) - integralRect.at<float>(rectSizey*0.25,rectSizex*0.75);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey,rectSizex*0.75) - integralRect.at<float>(rectSizey*0.25,rectSizex*0.75) - integralRect.at<float>(rectSizey,rectSizex*0.25) + integralRect.at<float>(rectSizey*0.25,rectSizex*0.25);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey*0.75,rectSizex*0.75) - integralRect.at<float>(rectSizey*0.75,rectSizex*0.25);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey*0.75,rectSizex*0.75) - integralRect.at<float>(rectSizey*0.25,rectSizex*0.75);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey*0.75,rectSizex) - integralRect.at<float>(rectSizey*0.25,rectSizex) - integralRect.at<float>(rectSizey*0.25,rectSizex*0.75) + integralRect.at<float>(rectSizey*0.25,rectSizex*0.25);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey-1,rectSizex-1);
						indx++;
						p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey*0.75,rectSizex);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey*0.25,rectSizex);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey,rectSizex) - integralRect.at<float>(rectSizey,rectSizex*0.75);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
						p = integralRect.at<float>(rectSizey,rectSizex*0.25);
						featureMat.at<float>(im, indx) = 2*p - integralRect.at<float>(rectSizey,rectSizex);
						indx++;
					}
				}
			}
		}

	}
	return featureMat;
}


Mat creatSumFeatures(vector<Mat*> trainingData, int trainingNum, int imageSize,bool test)
{
	cout << trainingData.size() << endl;
	if(test)
	{
		Mat featureMat = Mat::zeros(trainingData.size(),2,CV_32FC1);
		featureMat.at<int>(0,0) = 10;
		for(int im=0; im<trainingData.size(); im++)
		{
			featureMat.at<float>(im,1) = mean(*trainingData[im])(0);

		}
		return featureMat;
	}
	else
	{
		Mat featureMat = Mat::zeros(trainingData.size(),1,CV_32FC1);
		for(int im=0; im<trainingData.size(); im++)
		{
			featureMat.at<float>(im,0) = mean(*trainingData[im])(0);
			cout << featureMat.at<float>(im,0) << endl;
		}
		return featureMat;
	}
}