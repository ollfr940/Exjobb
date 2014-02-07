#include<string>
#include<iostream>
#include<fstream>
#include<math.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
using namespace std;
using namespace cv;

string intToStrIm(int i)
{
	string bla = "000000";
	stringstream s;
	stringstream ss;
	ss<< i%5;
	s<<(21 + i/5);
	string ret ="";
	string ret2;
	ss>>ret;
	s>>ret2;
	string name = bla.substr(0,bla.size()-ret.size());
	name = "img_T0002"+ret2+"_S"+name+ret+"_cpy.bmp";
	return name;
}

string intToStrGroundTruth(int i)
{
	string bla = "000000";
	stringstream s;
	stringstream ss;
	ss<<i%5;
	s<<(21 + i/5);
	string ret ="";
	string ret2;
	ss>>ret;
	s>>ret2;
	string name = bla.substr(0,bla.size()-ret.size());
	name = "img_T0002"+ret2+"_S"+name+ret+"_cpy_mask.png";
	return name;
}

void writeCvMatToFile(Mat& m, const char* filename)
{
	ofstream fout(filename);
	
	if(!fout)
	{
		cout<<"File Not Opened"<<endl;  return;
	}

	for(int i=0; i<m.rows; i++)
	{
		for(int j=0; j<m.cols-1; j++)
		{
			fout<<m.at<float>(i,j)<<"\t";
		}
		fout << m.at<float>(i,m.cols-1);
		fout<<endl;
	}

	fout.close();
}

void writeMatToFile(cv::Mat& m, vector<char> v, const char* filename)
{
	ofstream fout(filename);

	if(!fout)
	{
		cout<<"File Not Opened"<<endl;  return;
	}

	for(int i=0; i<m.rows; i++)
	{
		fout << v[i] << ',';
		for(int j=0; j<m.cols-1; j++)
		{
			fout<<m.at<float>(i,j)<<',';
		}
		fout << m.at<float>(i,m.cols-1);
		fout<<endl;
	}

	fout.close();
}

float calcStandardDeviation(cv::Mat& tile)
{
	Mat mean;
	Mat std;
	cv::meanStdDev(tile, mean, std);
	return std.at<double>(0,0);
}


Mat createLBPFeatures(int numberOfImages, int tileSize, int imageSize, int tileNum)
{
	printf("Calculate LBP features....\n\n");
	int features = 256;
	Mat featureMatrix = Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	Mat binary = Mat::zeros(tileSize-2,tileSize-2,CV_32FC1);
	int tiles = imageSize/tileSize, channels[] = {0, 1}, histSize[] = {256};
	float range[] = {0, 256};
	const float* ranges[] = {range};
	Mat im, tile, mean, std, block, LBPblock, bp;
	MatND hist;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = imread(intToStrIm(r), CV_LOAD_IMAGE_GRAYSCALE);
		
		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				meanStdDev(tile, mean, std);
				
				if(std.at<double>(0,0) < 20)
					featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = (tileSize-2)*(tileSize-2);
				else
				{
					for(int ii=0; ii<tileSize-2; ii++)
					{
						for(int jj=0; jj<tileSize-2; jj++)
						{
							block = tile(Rect(ii,jj,3,3));
							compare(block,block.at<uchar>(1,1),LBPblock,CMP_GT);
							LBPblock.convertTo(bp,CV_32FC1);
							binary.at<float>(ii,jj) = (bp.at<float>(0,0)+bp.at<float>(0,1)*2+bp.at<float>(0,2)*4+bp.at<float>(1,2)*8+
											bp.at<float>(2,2)*16+bp.at<float>(2,1)*32+bp.at<float>(2,0)*64+bp.at<float>(1,0)*128)/255;

							//cout << binary.at<float>(ii,jj) << endl;
						}
					}
					calcHist(&binary,1,channels,Mat(),hist,1,histSize,ranges,true,false);
					
					for(int f=0; f<features; f++)
						featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,f) = hist.at<float>(f);
				}
			}
		}
		cout << "image " << r+1 << endl;
	}
	return featureMatrix;
}

Mat createSimpleFeatures(int numberOfImages, int tileSize, int imageSize, int tileNum)
{
	printf("Calculate simple features....\n\n");
	int features = 2;
	Mat featureMatrix = Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	int tiles = imageSize/tileSize;
	Mat im, tile, harris, harrisNorm, tileharris, eigenVV;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = imread(intToStrIm(r), CV_LOAD_IMAGE_GRAYSCALE);
		cornerHarris(im,harris,2,3,0.05,BORDER_DEFAULT );
		//normalize(harris, harrisNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		//cv::cornerEigenValsAndVecs(im,eigenVV,tileSize,3,BORDER_DEFAULT);

		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				tileharris = harris(Rect(i*tileSize, j*tileSize, tileSize, tileSize));

				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = calcStandardDeviation(tile);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,1) = cv::sum(tileharris)(0);

			}
		}
		cout << "image " << r+1 << endl;
	}
	return featureMatrix;
}

vector<char> getResponses(int numberOfImages,int tileSize,int imageSize,int tileNum)
{
	printf("Calculate responses....\n\n");
	vector<char> responses;
	Mat groundTruth, groundTruthTile;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		groundTruth = imread(intToStrGroundTruth(r), CV_LOAD_IMAGE_GRAYSCALE);
		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				groundTruthTile = groundTruth(Rect(i*tileSize, j*tileSize, tileSize, tileSize));

				if(cv::sum(groundTruthTile)(0) > 150*tileSize*tileSize)
					responses.push_back('T');
				else
					responses.push_back('F');
			}
		}
	}
	return responses;
}