#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>
#include "OCRRecognize.h"
#include <iostream>
#include <time.h>
#include <fstream>

using namespace cv;

int main(int argc, char *argv[])
{
	OCRRecognize rec;

	//std::ofstream f("rec_res.txt");
	//if (!f)
	//{
	//	return -1;
	//}

	for (int i = 1; i <= 6; i++)
	{
		//f << "i: " << i << std::endl;
		cv::Mat reImg = imread("0" + std::to_string(i)+".jpg", 0);
		if (reImg.empty())
		{
			return -1;
		}
		char *pOutSrc = rec.recognize(reImg);
		std::cout << "pOutSrc: " << pOutSrc << std::endl;
		//f << "pOutSrc: " << pOutSrc;
		cv::imshow("src", reImg);

		////直方图均衡化
		//cv::Mat reImgCopy1;
		//reImg.copyTo(reImgCopy1);
		//cv::Rect rect1(0, 0, reImg.cols / 1, reImg.rows);
		//cv::equalizeHist(reImgCopy1(rect1), reImgCopy1(rect1));
		//cv::imshow("reImgCopy1", reImgCopy1);
		//char *pOut1 = rec.recognize(reImgCopy1);
		//std::cout << "pOut1: " << pOut1 << std::endl;
		
		////线性变化
		//cv::Mat reImgCopy2 = cv::Mat::zeros(reImg.size(), reImg.type());
		//reImg.copyTo(reImgCopy2);
		//cv::convertScaleAbs(reImgCopy2, reImgCopy2, 0.8, -20);
		//cv::normalize(reImgCopy2, reImgCopy2, 0, 255, NORM_MINMAX);
		//char *pOut2 = rec.recognize(reImgCopy2);
		//std::cout << "pOut2: " << pOut2 << std::endl;
		////f << "pOut2: " << pOut2;
		//cv::imshow("reImgCopy2", reImgCopy2);
		////cv::waitKey(0);
		
		//拉普拉斯增强
		cv::Mat reImgCopy3;
		reImg.copyTo(reImgCopy3);
		cv::Rect rect(0, 0, reImgCopy3.cols / 1, reImgCopy3.rows);

		cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
		cv::filter2D(reImg(rect), reImgCopy3(rect), CV_8UC1, kernel);

		char *pOut3 = rec.recognize(reImgCopy3);

		std::cout << "pOut3: " << pOut3 << std::endl;
		//f << "pOut3: " << pOut3;
		cv::imshow("reImgCopy3", reImgCopy3);

		//log处理
		cv::Mat reImgCopy4;
		reImg.copyTo(reImgCopy4);
		for (int i = 0; i < reImgCopy4.rows; i++)
		{
			for (int j = 0; j < reImgCopy4.cols; j++)
			{
				reImgCopy4.at<uchar>(i, j) = 18 * log(1 + (int)reImg.at<uchar>(i, j));
			}
		}
		cv::normalize(reImgCopy4, reImgCopy4, 0, 255, CV_MINMAX);
		cv::convertScaleAbs(reImgCopy4, reImgCopy4);

		char* pOut4 = rec.recognize(reImgCopy4);
		std::cout << "pOut4: " << pOut4 << std::endl;
		//f << "pOut4: " << pOut4;
		cv::imshow("reImgCopy4", reImgCopy4);

		//gamma处理
		cv::Mat reImgCopy5;
		reImg.copyTo(reImgCopy5);
		for (int i = 0; i < reImgCopy5.rows; i++)
		{
			for (int j = 0; j < reImgCopy5.cols; j++)
			{
				reImgCopy5.at<uchar>(i, j) = saturate_cast<uchar>(pow((float)reImgCopy5.at<uchar>(i, j) / 255, 5.0)*255.0);
			}
		}
		cv::normalize(reImgCopy5, reImgCopy5, 0, 255, CV_MINMAX);
		cv::convertScaleAbs(reImgCopy5, reImgCopy5);

		char* pOut5 = rec.recognize(reImgCopy5);
		std::cout << "pOut5: " << pOut5 << std::endl;
		//f << "pOut5: " << pOut5;
		cv::imshow("reImgCopy5", reImgCopy5);

		//add
		cv::Mat reImgCopy6 = (reImgCopy3 + reImgCopy4)/2;

		char* pOut6 = rec.recognize(reImgCopy6);
		std::cout << "pOut6: " << pOut6 << std::endl;
		//f << "pOut6: " << pOut6 << std::endl;
		cv::imshow("reImgCopy6", reImgCopy6);


		//分区域作处理
		cv::Mat reImgCopy7;
		reImg.copyTo(reImgCopy7);
		cv::Rect rect7_1(0, 0, reImgCopy7.cols / 2, reImgCopy7.rows);
		cv::Rect rect7_2(reImgCopy7.cols / 4 - 1, 0, reImgCopy7.cols / 2, reImgCopy7.rows);
		cv::Rect rect7_3(reImgCopy7.cols / 2 - 1, 0, reImgCopy7.cols / 2, reImgCopy7.rows);
		cv::Mat kernel1 = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
		cv::filter2D(reImg(rect7_1), reImgCopy7(rect7_1), CV_8UC1, kernel1);
		cv::imshow("reImgCopy7_1", reImgCopy7);
		char* pOut7_1 = rec.recognize(reImgCopy7);
		std::cout << "pOut7_1: " << pOut7_1 << std::endl;
		//cv::filter2D(reImg(rect7_2), reImgCopy7(rect7_2), CV_8UC1, kernel1);
		//cv::imshow("reImgCopy7_2", reImgCopy7);
		cv::filter2D(reImg(rect7_3), reImgCopy7(rect7_3), CV_8UC1, kernel1);
		cv::imshow("reImgCopy7_3", reImgCopy7);
		char* pOut7_2 = rec.recognize(reImgCopy7);
		std::cout << "pOut7_2: " << pOut7_2 << std::endl;

		cv::waitKey(0);
	}

	return 0;
}
