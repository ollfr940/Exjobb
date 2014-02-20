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

cv::Mat createStdFeatures(int numberOfImages,int firstImage, int tileSize, int imageSize, int tileNum)
{
	printf("Calculate Std features....\n\n");
	int features = 1;
	cv::Mat featureMatrix = cv::Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	int tiles = imageSize/tileSize;
	cv::Mat im, tile, mean, std;//, harris, harrisNorm, tileharris, eigenVV;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = cv::imread(intToStrIm(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);
		//cv::cornerHarris(im,harris,2,3,0.05,cv::BORDER_DEFAULT );
		//normalize(harris, harrisNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
		//cv::cornerEigenValsAndVecs(im,eigenVV,tileSize,3,BORDER_DEFAULT);

		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				//tileharris = harris(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));

				cv::meanStdDev(tile, mean, std);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = (float)std.at<double>(0,0);
				//featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,1) = cv::sum(tileharris)(0);

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
	float score1D;
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
				score1D = cv::pow((eigenvalues[0]-eigenvalues[1]),2)/(pow(eigenvalues[0],2)+pow(eigenvalues[1],2));

				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = mean.at<double>(0,0);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,1) = std.at<double>(0,0);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,2) = score1D;
			}
		}
		std::cout << "image " << r+1 << std::endl;
	}
	return featureMatrix;
}

cv::Mat createFastCornerFeatures(int numberOfImages,int firstImage, int tileSize, int imageSize, int tileNum)
{
	printf("Calculate FAST corner detection features....\n\n");
	int features = 3;
	cv::Mat featureMatrix = cv::Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	int tiles = imageSize/tileSize;
	cv::Mat im, tile;
	std::vector<cv::KeyPoint> keyPoint;

	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = cv::imread(intToStrIm(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);

		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				cv::FAST(tile,keyPoint,10,true);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = keyPoint.size();
				cv::FAST(tile,keyPoint,25,true);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,1) = keyPoint.size();
				cv::FAST(tile,keyPoint,50,true);
				featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,2) = keyPoint.size();
			}
		}
		std::cout << "image " << r+1 << std::endl;
	}
	return featureMatrix;
}

cv::Mat createORBFeatures(int numberOfImages,int firstImage, int tileSize, int imageSize, int tileNum)
{
	printf("Calculate FAST corner detection features....\n\n");
	int features = 3;
	cv::Mat featureMatrix = cv::Mat::zeros(numberOfImages*tileNum*tileNum,features, CV_32FC1);
	int tiles = imageSize/tileSize;
	cv::Mat im, tile, descriptors, mean, std;
	std::vector<cv::KeyPoint> keyPoint;
	cv::ORB orb;
	
	for(int r = 0 ; r< numberOfImages ; r++)
	{
		im = cv::imread(intToStrIm(r+firstImage), CV_LOAD_IMAGE_GRAYSCALE);

		for(int i=0; i<tileNum; i++)
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
				
				orb(tile,cv::Mat(),keyPoint,descriptors);
				if(keyPoint.size())
				std::cout << keyPoint.size() << " " << descriptors.size() << std::endl;
				//featureMatrix.at<float>(r*tileNum*tileNum+i*tileNum+j,0) = keyPoint.size();
				
			}
		}
		std::cout << "image " << r+1 << std::endl;
	}
	return featureMatrix;
}

std::vector<char> getResponses1D(int numberOfImages,int firstImage,int tileSize,int imageSize,int tileNum, bool training)
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

				if(training)
				{
					if(cv::sum(groundTruthTile)(0) <= 100*tileSize*tileSize && cv::countNonZero(groundTruthTile) > 0.75*tileSize*tileSize)
						responses.push_back('T');
					else
						responses.push_back('F');
				}
				else
				{
					if(cv::sum(groundTruthTile)(0) <= 100*tileSize*tileSize && cv::countNonZero(groundTruthTile) > 0.75*tileSize*tileSize)
						responses.push_back('T');
					else if(cv::countNonZero(groundTruthTile) == 0)
						responses.push_back('F');
					else if(cv::sum(groundTruthTile)(0)/countNonZero(groundTruthTile) > 150)
						responses.push_back('F');
					else
						responses.push_back('I');
				}
			}
		}
	}
	return responses;
}

std::vector<char> getResponses2D(int numberOfImages,int firstImage,int tileSize,int imageSize,int tileNum,bool training)
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

				if(training)
				{
					if(cv::sum(groundTruthTile)(0) > 150*tileSize*tileSize)
						responses.push_back('T');
					else
						responses.push_back('F');
				}
				else
				{
					if(cv::sum(groundTruthTile)(0) > 150*tileSize*tileSize)
						responses.push_back('T');
					else if(cv::countNonZero(groundTruthTile) == 0)
						responses.push_back('F');
					else if(cv::sum(groundTruthTile)(0)/countNonZero(groundTruthTile) < 150)
						responses.push_back('F');
					else
						responses.push_back('I');
				}
			}
		}
	}
	return responses;
}

std::vector<char> getResponses(int numberOfImages,int firstImage,int tileSize,int imageSize,int tileNum,bool training, std::string& codeType)
{
	if(codeType == "1D")
		printf("Calculate groundtruth for 1D codes....\n\n");
	else if(codeType == "2D")
		printf("Calculate groundtruth for 2D codes....\n\n");
	else
		printf("Calculte groundtruth for 1D and 2D codes....\n\n");

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

				if(codeType == "1D")
				{
					if(training)
					{
						if(cv::sum(groundTruthTile)(0) <= 100*tileSize*tileSize && cv::countNonZero(groundTruthTile) > 0.75*tileSize*tileSize)
							responses.push_back('T');
						else
							responses.push_back('F');
					}
					else
					{
						if(cv::sum(groundTruthTile)(0) <= 100*tileSize*tileSize && cv::countNonZero(groundTruthTile) > 0.75*tileSize*tileSize)
							responses.push_back('T');
						else if(cv::countNonZero(groundTruthTile) == 0)
							responses.push_back('F');
						else if(cv::sum(groundTruthTile)(0)/countNonZero(groundTruthTile) > 150)
							responses.push_back('F');
						else
							responses.push_back('I');
					}
				}
				else if(codeType == "2D")
				{
					if(training)
					{
						if(cv::sum(groundTruthTile)(0) > 150*tileSize*tileSize)
							responses.push_back('T');
						else
							responses.push_back('F');
					}
					else
					{
						if(cv::sum(groundTruthTile)(0) > 150*tileSize*tileSize)
							responses.push_back('T');
						else if(cv::countNonZero(groundTruthTile) == 0)
							responses.push_back('F');
						else if(cv::sum(groundTruthTile)(0)/countNonZero(groundTruthTile) < 150)
							responses.push_back('F');
						else
							responses.push_back('I');
					}
				}
				else
				{
					if(training)
					{
						if(cv::sum(groundTruthTile)(0) > 150*tileSize*tileSize)
							responses.push_back('T');
						else if(cv::sum(groundTruthTile)(0) <= 100*tileSize*tileSize && cv::countNonZero(groundTruthTile) > 0.75*tileSize*tileSize)
							responses.push_back('T');
						else
							responses.push_back('F');
					}
					else
					{
						if(cv::sum(groundTruthTile)(0) > 150*tileSize*tileSize)
							responses.push_back('T');
						else if(cv::sum(groundTruthTile)(0) <= 100*tileSize*tileSize && cv::countNonZero(groundTruthTile) > 0.75*tileSize*tileSize)
							responses.push_back('T');
						else if(cv::countNonZero(groundTruthTile) == 0)
							responses.push_back('F');
						else
							responses.push_back('I');
					}
				}
			}
		}
	}
	return responses;
}

void evaluateResult(int firstImage,int k, int scaleDown,int imNum, int tileSize, int imageSize, int tileNum, CvBoost boost, cv::Mat& featureMat,std::vector<char>& responses, float* trueClass, float* falseClass,float strongClassThres)
{

	int features = featureMat.cols, imageCount=0, response;
	int imPos = tileSize/scaleDown;
	double totalTrueReal = 0, totalTrueClass = 0, totalFalseClass = 0;

	CvSeq* weights = boost.get_weak_predictors();
	int numOfWeights = weights->total;
	cv::Mat testSample(1,features+1, CV_32FC1 );
	cv::Mat Im(imageSize,imageSize,16);
	cv::Mat resizedIm(imageSize/scaleDown,imageSize/scaleDown,16);
	CvMat* weakResponses = cvCreateMat(1,numOfWeights,CV_32FC1);

	for(int i=0; i<imNum; i++)
	{
		trueClass[i] = 0;
		falseClass[i] = 0;
		int trueReal = 0;
		cv::Mat im = cv::imread(intToStrIm(firstImage+i));

		if(i >= k && i<scaleDown*scaleDown+k)
			cv::resize(im,resizedIm,cv::Size(imageSize/scaleDown,imageSize/scaleDown));

		for( int y = 0; y < tileNum*tileNum; y++ )
		{
			testSample.at<float>(0,0) = responses[y+i*tileNum*tileNum];
			for(int x=1; x < features+1; x++)
			{

				testSample.at<float>(0,x) = featureMat.at<float>(y+i*tileNum*tileNum,x-1);
			}

			if(strongClassThres == 0)
				response = (int)boost.predict( testSample);
			else
			{
				CvMat testSampleCvMat = testSample;
				boost.predict(&testSampleCvMat,(const CvMat*)0, weakResponses);
				float weakSum = cvSum(weakResponses).val[0];

				if(weakSum > strongClassThres)
					response = 2;
				else
					response = 1;
			}

			if(i >= k && i<scaleDown*scaleDown+k)
			{
				if(response == 2 && responses[y+i*tileNum*tileNum] != 'F')
					rectangle(resizedIm,cvPoint(y/tileNum*imPos,y%tileNum*imPos),cvPoint(y/tileNum*imPos + imPos,y%tileNum*imPos+imPos),CV_RGB(0,255,0),1,8);

				else if(responses[y+i*tileNum*tileNum] == 'T')
					rectangle(resizedIm,cvPoint(y/tileNum*imPos,y%tileNum*imPos),cvPoint(y/tileNum*imPos + imPos,y%tileNum*imPos+imPos),CV_RGB(255,0,0),1,8);

				else if(response == 2)
					rectangle(resizedIm,cvPoint(y/tileNum*imPos,y%tileNum*imPos),cvPoint(y/tileNum*imPos + imPos,y%tileNum*imPos+imPos),CV_RGB(0,0,255),1,8);
			}

			//for(int j=0; j<100; j++)
			//std::cout << cvmGet(weakResponses,0,j) << std::endl << cvSum(weakResponses).val[0] << std::endl;
			if(responses[y+i*tileNum*tileNum] == 'T')
				trueReal++;

			if(response == 2 && responses[y+i*tileNum*tileNum] == 'T')
				trueClass[i]++;

			if(response == 2 && responses[y+i*tileNum*tileNum] == 'F')
				falseClass[i]++;


		}
		//std::cout << i << std::endl;
		if(i >= k && i<scaleDown*scaleDown+k)
		{
			resizedIm.copyTo(Im(cv::Rect(imageCount/scaleDown*(imageSize/scaleDown),imageCount%scaleDown*(imageSize/scaleDown),imageSize/scaleDown,imageSize/scaleDown)));
			imageCount++;
		}

		totalTrueReal += trueReal;
		totalTrueClass += trueClass[i];
		totalFalseClass += falseClass[i];

		if(trueReal > 0)
			trueClass[i] = trueClass[i]/trueReal*100;
		else
			trueClass[i] = -20;
	}
	std::cout << "Amount of true tiles detected: " << totalTrueClass/totalTrueReal << std::endl
		<< "Average false tiles per image: " << totalFalseClass/imNum << std::endl;

	cv::namedWindow("Test images",CV_WINDOW_AUTOSIZE);
	imshow("Test images", Im);

	PlotManager pm;
	pm.Plot("Result from testing", trueClass  , imNum, 1,0,0,255);
	pm.Plot("Result from testing", falseClass  , imNum, 1,255,0,0);

	cv::waitKey();
	cv::destroyWindow("Test images");
}

void visualizeFeature(int image,int imageSize, int tileSize, int tileNum, int feature, cv::Mat& featureData)
{
	cv::Mat Im = cv::Mat::zeros(imageSize/2,imageSize/2, CV_32FC1);
	cv::Mat im = cv::imread(intToStrIm(image),CV_LOAD_IMAGE_GRAYSCALE);
	cv::resize(im,im,cv::Size(imageSize/2,imageSize/2));

	for(int i=0; i<tileNum*tileNum; i++)
		Im(cv::Rect(i/tileNum*(tileSize/2),i%tileNum*(tileSize/2),tileSize/2,tileSize/2)) = featureData.at<float>(image*tileNum*tileNum + i,feature);

	cv::normalize(Im, Im, 0, 255, cv::NORM_MINMAX,CV_8UC1);
	cv::namedWindow("image",CV_WINDOW_AUTOSIZE);
	imshow("image", im);

	cv::namedWindow("Test image",CV_WINDOW_AUTOSIZE);
	imshow("Test image", Im);
	cv::waitKey();
	cv::destroyWindow("Test image");
}

void cascade(int firstImage,int imNum, int tileSize, int imageSize, int tileNum, std::vector<CvBoost*> boost,std::vector<char>& responses,std::vector<float> strongClassThres)
{
	cv::Mat im, tile, mean, std;
	int response, features, cascadeStep;
	std::vector<int> weights;
	std::vector<CvMat*> weakResponses;
	std::vector<cv::KeyPoint> keyPoint;

	for(int i=0; i<boost.size(); i++)
		weakResponses.push_back(cvCreateMat(1,boost[i]->get_weak_predictors()->total,CV_32FC1));
	
	//CvMat* weakResponses = cvCreateMat(1,numOfWeights,CV_32FC1);

	for(int r=0; r<imNum; r++)
	{

		im = cv::imread(intToStrIm(firstImage+r));

		for( int i=0; i<tileNum; i++ )
		{
			for(int j=0; j<tileNum; j++)
			{
				tile = im(cv::Rect(i*tileSize, j*tileSize, tileSize, tileSize));
		

			//Calculate std
				cascadeStep = 0;
				features = 1;
				cv::Mat testSampleSTD(1,features+1, CV_32FC1 );
				testSampleSTD.at<float>(0,0) = responses[r*tileNum*tileNum + i*tileNum + j]; 
				cv::meanStdDev(tile,mean,std);
				testSampleSTD.at<float>(0,1) = (float)std.at<double>(0,0);

				CvMat testSampleCvMat = testSampleSTD;
				boost[cascadeStep]->predict(&testSampleCvMat,(const CvMat*)0, weakResponses[cascadeStep]);
				float weakSum = cvSum(weakResponses[cascadeStep]).val[cascadeStep];

				if(weakSum > strongClassThres[cascadeStep])
				{
					// Calculate FAST corners
					cascadeStep = 1;
					features = 3;
					cv::Mat testSampleFAST(1,features+1, CV_32FC1 );
					testSampleFAST.at<float>(0,0) = responses[r*tileNum*tileNum + i*tileNum + j]; 

					cv::FAST(tile,keyPoint,10,true);
					testSampleFAST.at<float>(0,1) = keyPoint.size();
					cv::FAST(tile,keyPoint,25,true);
					testSampleFAST.at<float>(0,2) = keyPoint.size();
					cv::FAST(tile,keyPoint,50,true);
					testSampleFAST.at<float>(0,3) = keyPoint.size();

					CvMat testSampleCvMat = testSampleSTD;
					boost[cascadeStep]->predict(&testSampleCvMat,(const CvMat*)0, weakResponses[cascadeStep]);
					float weakSum = cvSum(weakResponses[cascadeStep]).val[cascadeStep];

					if(weakSum > strongClassThres[cascadeStep])
					{
						// Calculate LBP
						cascadeStep = 2;
						features = 256;

					}
				}
			}
		}
		std::cout << "image " << r << std::endl;
	}
}