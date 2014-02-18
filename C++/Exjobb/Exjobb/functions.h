#include<string>
#include<iostream>
#include<fstream>
#include<math.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
#include "cvplot.h"


std::string intToStrIm(int i)
{
	std::string bla = "000000";
	std::stringstream s;
	std::stringstream ss;
	ss<< i%5;
	s<<(21 + i/5);
	std::string ret ="";
	std::string ret2;
	ss>>ret;
	s>>ret2;
	std::string name = bla.substr(0,bla.size()-ret.size());
	name = "img_T0002"+ret2+"_S"+name+ret+"_cpy.bmp";
	return name;
}

std::string intToStrGroundTruth(int i)
{
	std::string bla = "000000";
	std::stringstream s;
	std::stringstream ss;
	ss<<i%5;
	s<<(21 + i/5);
	std::string ret ="";
	std::string ret2;
	ss>>ret;
	s>>ret2;
	std::string name = bla.substr(0,bla.size()-ret.size());
	name = "img_T0002"+ret2+"_S"+name+ret+"_cpy_mask.png";
	return name;
}

void writeMatToXML(cv::Mat& m,const char* filename)
{
	printf("Saving test data....\n\n");
	cv::FileStorage fs(filename, cv::FileStorage::WRITE );
	fs << "testData" << m; 
	fs.release();
}

cv::Mat readMatFromXML(const char* filename)
{
	printf("Reading test data....\n\n");
	cv::Mat m;
	cv::FileStorage fs(filename, cv::FileStorage::READ );
	fs["testData"] >> m;
	return m;
}

void writeFileToMatlab(cv::Mat& m, const char* filename)
{
	std::ofstream fout(filename);

	if(!fout)
	{
		std::cout<<"File Not Opened"<<std::endl;  return;
	}

	for(int i=0; i<m.rows; i++)
	{
		for(int j=0; j<m.cols; j++)
		{
			fout<<m.at<float>(i,j)<<' ';
		}
		fout<<std::endl;
	}

	fout.close();
}


void writeMatToFile(cv::Mat& m, std::vector<char> v, const char* filename)
{
	std::ofstream fout(filename);

	if(!fout)
	{
		std::cout<<"File Not Opened"<<std::endl;  return;
	}

	for(int i=0; i<m.rows; i++)
	{
		fout << v[i] << ',';
		for(int j=0; j<m.cols-1; j++)
		{
			fout<<m.at<float>(i,j)<<',';
		}
		fout << m.at<float>(i,m.cols-1);
		fout<<std::endl;
	}

	fout.close();
}

float calcStandardDeviation(cv::Mat& tile)
{
	cv::Mat mean;
	cv::Mat std;
	cv::meanStdDev(tile, mean, std);
	return std.at<double>(0,0);
}


cv::Mat createLBPFeatures(int numberOfImages,int firstImage, int tileSize, int imageSize, int tileNum)
{
	printf("Calculate LBP features....\n\n");
	int features = 256;
	cv::Mat featureMatrix = cv::Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	cv::Mat binary = cv::Mat::zeros(tileSize-2,tileSize-2,CV_32FC1);
	int tiles = imageSize/tileSize, channels[] = {0, 1}, histSize[] = {256};
	float range[] = {0, 256};
	const float* ranges[] = {range};
	cv::Mat im, tile, mean, std, block, LBPblock, bp;
	cv::MatND hist;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = cv::imread(intToStrIm(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);

		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				meanStdDev(tile, mean, std);

				if(std.at<double>(0,0) < 20)
					featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = (tileSize-2)*(tileSize-2);
				else
				{
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

					for(int f=0; f<features; f++)
						featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,f) = hist.at<float>(f);
				}
			}
		}
		std::cout << "image " << r+1 << std::endl;
	}
	return featureMatrix;
}

cv::Mat createSimpleFeatures(int numberOfImages,int firstImage, int tileSize, int imageSize, int tileNum)
{
	printf("Calculate simple features....\n\n");
	int features = 2;
	cv::Mat featureMatrix = cv::Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	int tiles = imageSize/tileSize;
	cv::Mat im, tile, harris, harrisNorm, tileharris, eigenVV;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = cv::imread(intToStrIm(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);
		cv::cornerHarris(im,harris,2,3,0.05,cv::BORDER_DEFAULT );
		//normalize(harris, harrisNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		//cv::cornerEigenValsAndVecs(im,eigenVV,tileSize,3,BORDER_DEFAULT);

		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				tileharris = harris(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));

				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = calcStandardDeviation(tile);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,1) = cv::sum(tileharris)(0);

			}
		}
		std::cout << "image " << r+1 << std::endl;
	}
	return featureMatrix;
}

cv::Mat createDistanceFeatures(int numberOfImages,int firstImage, int tileSize, int imageSize, int tileNum,int threshold1,int threshold2)
{
	printf("Calculate Distance features....\n\n");
	int features = 3;
	cv::Mat featureMatrix = cv::Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	int tiles = imageSize/tileSize;
	int score1D;
	cv::Mat im, distanceTile, sobxTile, sobyTile , canny, canny2, distanceMap, mean, std, sobx, soby;
	cv::Mat structureTensor = cv::Mat::zeros(2,2,CV_32FC1);
	std::vector<float> eigenvalues;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = cv::imread(intToStrIm(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);
		cv::Canny(im,canny,threshold1,threshold2);
		cv::threshold(canny,canny,128,255, cv::THRESH_BINARY_INV);
		cv::distanceTransform(canny,distanceMap,CV_DIST_L1,3);
		
		cv::GaussianBlur( im, im, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
		cv::Sobel( im, sobx, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );
		//cv::convertScaleAbs( sobx, gradx);
		cv::Sobel(im, soby, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT );


		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				distanceTile = distanceMap(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				sobxTile = sobx(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				sobyTile = soby(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));

				cv::meanStdDev(distanceTile, mean, std);
				structureTensor.at<float>(0,0) = sobxTile.dot(sobxTile);
				structureTensor.at<float>(0,1) = sobxTile.dot(sobyTile);
				structureTensor.at<float>(1,0) = sobxTile.dot(sobyTile);
				structureTensor.at<float>(1,1) = sobyTile.dot(sobyTile);
				cv::eigen(structureTensor, eigenvalues);
				score1D = cv::pow((eigenvalues[0]-eigenvalues[1]),2)/(pow(eigenvalues[0],2)*pow(eigenvalues[1],2));

				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = mean.at<double>(0,0);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,1) = std.at<double>(0,0);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,2) = score1D;
			}
		}
		std::cout << "image " << r+1 << std::endl;
	}
	return featureMatrix;
}

/*cv::Mat createFastCornerFeatures(int numberOfImages,int firstImage, int tileSize, int imageSize, int tileNum,int threashold)
{
	printf("Calculate FAST corner detection features....\n\n");
	int features = 16;
	cv::Mat featureMatrix = cv::Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	int tiles = imageSize/tileSize;
	cv::Mat im, tile;
	cv::FastFeatureDetector fast;
	//std::vector<cv::KeyPoint,std::allocator<cv::KeyPoint>> keyPoint;
	std::vector<cv::KeyPoint> keyPoint;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = cv::imread(intToStrIm(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);
		fast.detect(im,keyPoint);
		
		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				cv::FAST(tile,keyPoint,threashold,true);
				std::cout << keyPoint.size() << std::endl;
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = calcStandardDeviation(tile);


			}
		}
		std::cout << "image " << r+1 << std::endl;
	}
	return featureMatrix;
}
*/
std::vector<char> getResponses1D(int numberOfImages,int firstImage,int tileSize,int imageSize,int tileNum)
{
	printf("Calculate 1D responses....\n\n");
	std::vector<char> responses;
	cv::Mat groundTruth, groundTruthTile;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		groundTruth = cv::imread(intToStrGroundTruth(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);
		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				groundTruthTile = groundTruth(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				if(cv::sum(groundTruthTile)(0) <= 100*tileSize*tileSize && cv::countNonZero(groundTruthTile) > 0.75*tileSize*tileSize)
					responses.push_back('T');
				else
					responses.push_back('F');
			}
		}
	}
	return responses;
}

std::vector<char> getResponses2D(int numberOfImages,int firstImage,int tileSize,int imageSize,int tileNum)
{
	printf("Calculate 2D responses....\n\n");
	std::vector<char> responses;
	cv::Mat groundTruth, groundTruthTile;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		groundTruth = cv::imread(intToStrGroundTruth(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);
		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				groundTruthTile = groundTruth(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));

				if(cv::sum(groundTruthTile)(0) > 150*tileSize*tileSize)
					responses.push_back('T');
				else
					responses.push_back('F');
			}
		}
	}
	return responses;
}

void testImages(int firstImage,int k, int scaleDown, int tileSize, int imageSize, int tileNum, CvBoost& boost, cv::Mat& featureMat,std::vector<char>& responses)
{
	int features = featureMat.cols;
	int imPos = tileSize/scaleDown;

	cv::Mat testSample(1,features+1, CV_32FC1 );
	cv::Mat Im(imageSize,imageSize,16);
	cv::Mat resizedIm(imageSize/scaleDown,imageSize/scaleDown,16);


	for(int i=0; i<scaleDown*scaleDown; i++)
	{
		cv::Mat im = cv::imread(intToStrIm(firstImage+k+i));
		resize(im,resizedIm,cv::Size(imageSize/scaleDown,imageSize/scaleDown));

		for( int y = 0; y < tileNum*tileNum; y++ )
		{
			testSample.at<float>(0,0) = responses[y+(i+k)*tileNum*tileNum];
			for(int x=1; x < features+1; x++)
			{

				testSample.at<float>(0,x) = featureMat.at<float>(y+(i+k)*tileNum*tileNum,x-1);
			}
			int response = (int)boost.predict( testSample); //,cv::Mat(),cv::Range::all(),true);

			if(response == 2)
				rectangle(resizedIm,cvPoint(y/tileNum*imPos,y%tileNum*imPos),cvPoint(y/tileNum*imPos + imPos,y%tileNum*imPos+imPos),CV_RGB(0,255,0),1,8);
			else if(responses[y+(i+k)*tileNum*tileNum] == 'T')
				rectangle(resizedIm,cvPoint(y/tileNum*imPos,y%tileNum*imPos),cvPoint(y/tileNum*imPos + imPos,y%tileNum*imPos+imPos),CV_RGB(255,0,0),1,8);
		}
		resizedIm.copyTo(Im(cv::Rect(i/scaleDown*(imageSize/scaleDown),i%scaleDown*(imageSize/scaleDown),imageSize/scaleDown,imageSize/scaleDown)));

	}
	cv::namedWindow("Test image",CV_WINDOW_AUTOSIZE);
	imshow("Test image", Im);
	cv::waitKey();
	cv::destroyWindow("Test image");
}

void testForPlot(int firstImage,int imNum, int tileSize, int imageSize, int tileNum, CvBoost boost, cv::Mat& featureMat,std::vector<char>& responses, float* trueClass, float* falseClass)
{
	int features = featureMat.cols;
	double totalTrueReal = 0, totalTrueClass = 0;
	for(int i=0; i<imNum; i++)
	{
		trueClass[i] = 0;
		falseClass[i] = 0;
		int trueReal = 0;
		cv::Mat testSample(1,features+1, CV_32FC1 );

		for( int y = 0; y < tileNum*tileNum; y++ )
		{
			testSample.at<float>(0,0) = responses[y+i*tileNum*tileNum];
			for(int x=1; x < features+1; x++)
			{

				testSample.at<float>(0,x) = featureMat.at<float>(y+i*tileNum*tileNum,x-1);
			}
			float response = (int)boost.predict( testSample, cv::Mat(),cv::Range::all(),true,false);
			 std::cout << response << std::endl;
			if(responses[y+i*tileNum*tileNum] == 'T')
				trueReal++;

			if(response == 2 && responses[y+i*tileNum*tileNum] == 'T')
				trueClass[i]++;

			if(response == 2 && responses[y+i*tileNum*tileNum] == 'F')
				falseClass[i]++;
		}

		totalTrueReal += trueReal;
		totalTrueClass += trueClass[i];

		if(trueReal > 0)
			trueClass[i] = trueClass[i]/trueReal*100;
		else
			trueClass[i] = -20;

	}
	std::cout << "Amount of true tiles detected: " << totalTrueClass/totalTrueReal << std::endl;
	PlotManager pm;
	pm.Plot("True detection in percent", trueClass  , imNum, 1,0,0,255);
	pm.Plot("True detection in percent", falseClass  , imNum, 1,255,0,0);
	cv::waitKey();
}

void visualizeFeature(int image,int imageSize, int tileNum, int feature, cv::Mat& featureData)
{
	cv::Mat featureIm = cv::Mat::zeros(tileNum,tileNum,CV_32FC1);
	cv::Mat im = cv::imread(intToStrIm(image),CV_LOAD_IMAGE_GRAYSCALE);
	cv::resize(im,im,cv::Size(imageSize/2,imageSize/2));
	std::cout << featureData.size() << std::endl;

	for(int i=0; i<tileNum*tileNum; i++)
	{
		featureIm.at<float>(i/tileNum,i%tileNum) = featureData.at<float>(image*tileNum*tileNum + i,feature);
	}
	cv::resize(featureIm,featureIm,cv::Size(imageSize/2,imageSize/2));
	cv::normalize(featureIm, featureIm, 0.0, 1.0, cv::NORM_MINMAX,CV_8UC1);
	cv::equalizeHist(featureIm,featureIm);
	cv::namedWindow("image",CV_WINDOW_AUTOSIZE);
	imshow("image", im);

	cv::namedWindow("Test image",CV_WINDOW_AUTOSIZE);
	imshow("Test image", featureIm);
	cv::waitKey();
	cv::destroyWindow("Test image");
}