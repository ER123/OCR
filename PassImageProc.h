#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>
#include <string>
#include <vector>
#include <algorithm>

class PassImageProc
{
public:
	PassImageProc();
	PassImageProc(cv::Mat srcImg);
	~PassImageProc();
	cv::Mat getROIImage();
	void imgProc(cv::Mat &srcImg, const cv::Size &closeSize = cv::Size(17, 5), const cv::Size &erSize = cv::Size(3, 3), const double binaryMinVal = 200);
	void imgProc2(cv::Mat &srcImg, const cv::Size &closeSize = cv::Size(17, 5), const cv::Size &erSize = cv::Size(3, 3), const double binaryMinVal = 200);
	void imgProc1(cv::Mat &srcImg, const cv::Size &closeSize = cv::Size(17, 5), const cv::Size &erSize = cv::Size(3, 3), const double binaryMinVal = 200);
	std::vector<cv::Mat> getResultImageArray();
	static void putTextZH(cv::Mat &dst, const char* str, cv::Point org, cv::Scalar color, int fontSize, const char* fn, bool italic, bool underline);
	static std::vector<cv::Mat> mResultImgs;

private:
	cv::Mat mSrcImg;
	std::vector<cv::Mat> mProcImgs;
	void unevenLightCompensate(cv::Mat &image, int blockSize);
	static void GetStringSize(HDC hDC, const char* str, int* w, int* h);
	std::vector<cv::RotatedRect> findTextRegion(cv::Mat, cv::Mat);
	cv::Mat ImgRotate(const cv::Mat& ucmatImg, double dDegree);
	cv::Mat normalizedMatByRoi(const cv::Mat &cpsrcMat, const cv::RotatedRect &rotatedRect);
	cv::Mat image_smoothening(cv::Mat &roiImage, double thresholdValue);
	cv::Mat remove_noise_and_smooth(cv::Mat &img);
	bool verifySize(cv::RotatedRect mr);

	bool hasPoints(std::vector<cv::Point>, std::vector<std::vector<cv::Point>>);///检查直线是否平行重合
	std::vector<cv::Point2f> findCorners(std::vector<std::vector<cv::Point>> points, int w, int h);      ///找到直线的四个交点
	std::vector<cv::Point2f> sortCorner(std::vector<cv::Point2f>);
	//bool setSortRule(const cv::Point2f& , const cv::Point2f& );
	bool isEligible(const cv::RotatedRect &);
	cv::Mat image_rotate_newsize(cv::Mat&, const CvPoint &, double, double);
	cv::Mat findCardRegion(cv::Mat, int ,float scale = 0.5);
};

