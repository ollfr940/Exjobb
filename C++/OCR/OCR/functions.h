#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include<fstream>
#include<iostream>

void saveInfo(std::vector<std::string> info,const char* filename)
{
	std::ofstream fout(filename);

	if(!fout)
	{
		std::cout<<"File Not Opened"<<std::endl; 
		return;
	}

	for(int i=0; i<info.size(); i++)
		fout << info[i] << std::endl;

	fout.close();
}


std::string intToStr(int i,char l)
{
	std::string bla = "0000";
	std::stringstream ss;
	ss<< i;
	std::string ret ="";
	ss>>ret;
	std::string name = bla.substr(0,bla.size()-ret.size());
	name = l + name+ret+".bmp";

	return name;
}

int Displaying_Random_Text(cv::RNG rng, char l,int letNum,int imNum)
{
	int lineType = 8;
	char let = 'A';
	std::string letstr;
	std::string* name;
	std::vector<std::string> info;
	for(int r = 0; r < imNum; r++)
	{
		cv::Mat image = cv::imread("white.png");
		for ( int i = 0; i < letNum; i++ )
		{
			if(let == l && let != 'Z')
				let++;
			else if(let == 'Z' && l == 'Z')
				let = 'A';

			cv::Point org;
			org.x = rng.uniform(0, 1358);
			org.y = rng.uniform(0, 789);
			letstr = let;
			cv::putText( image, letstr, org, 0,//rng.uniform(0,8),
				rng.uniform(30,100)*0.05+0.1, 0, rng.uniform(4,7), lineType);

			if(let == 'Z')
				let = 'A';
			else
				let++;
		}
		name = new std::string(intToStr(r,l));
		cv::imwrite(*name,image);
		info.push_back(*name);

		//cv::imshow(*name,image);
		//cv::waitKey();
		//cv::destroyWindow(*name);
	}
	saveInfo(info,"info.txt");

	return 0;
}
