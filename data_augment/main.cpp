#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include<ctime>

#include "image_augmentor.h"

using namespace std;

int main(){
  ImageAugmentor image_augmentor = new ImageAugmentor();
  cv::Mat mat = cv::imread("test.png");
  cv::namedWindow("main");
  cv::imshow("main", mat);
  cv::waitKey();

  double start = clock();
  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.SimulateIllumination(mat, i % 4));
    cv::waitKey();
    // image_augmentor.SimulateIllumination(mat, i % 4);
  }
  double end = clock();
  cout << "Use time: " << end - start << endl;
  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.ColorJittering(mat, false));
    cv::waitKey();
  }

  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.RotateImage(mat, true));
    cv::waitKey();
  }

  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.AddPepperSaltNoisy(mat, false));
    cv::waitKey();
    cv::imshow("main", image_augmentor.AddGaussianNoisy(mat, false));
    cv::waitKey();
  }

  for(int i =0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.ScaleUpDown(mat, 0.30, 0.86, 5, 1));
    cv::waitKey();
  }

  for(int i = -1; i < 2; ++i) {
    cv::imshow("main", image_augmentor.FlipImage(mat, i));
    cv::waitKey();
  }
  return 0;
}
