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
  cv::Mat mat = cv::imread("1.png");
  cv::namedWindow("main");
  cv::imshow("main", mat);
  cv::waitKey();

  string extension = ".png";
  string save_dir = "./aug/";

  double start = clock();
  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.SimulateIllumination(mat, i % 4));
    string filename = save_dir + "sm" + (char)(i + '0') + extension;
    cv::imwrite(filename, image_augmentor.SimulateIllumination(mat, i % 4));
    cv::waitKey();
    // image_augmentor.SimulateIllumination(mat, i % 4);
  }
  double end = clock();
  cout << "Use time: " << end - start << endl;
  // string extension = ".png";
  // string save_dir = "./data/";
  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.ColorJittering(mat, false));
    string filename = save_dir + "cj" + (char)(i + '0') + extension;
    cv::imwrite(filename, image_augmentor.ColorJittering(mat, false));
    cv::waitKey();
  }

  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.RotateImage(mat, true));
    string filename = save_dir + "rt" + (char)(i + '0') + extension;
    cv::imwrite(filename, image_augmentor.RotateImage(mat, false));
    cv::waitKey();
  }

  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.AddPepperSaltNoisy(mat, false));
    string filename1 = save_dir + "ps" + (char)(i + '0') + extension;
    string filename2 = save_dir + "gs" + (char)(i + '0') + extension;    
    cv::imwrite(filename1, image_augmentor.AddPepperSaltNoisy(mat, false));
    cv::waitKey();
    cv::imshow("main", image_augmentor.AddGaussianNoisy(mat, false));
    cv::imwrite(filename2, image_augmentor.AddGaussianNoisy(mat, false));
    cv::waitKey();
  }

  for(int i =0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.ScaleUpDown(mat, 0.30, 0.86, 5, 1));
    string filename = save_dir + "ud" + (char)(i + '0') + extension;
    cv::imwrite(filename, image_augmentor.ScaleUpDown(mat, 0.30, 0.86, 5, 1));
    cv::waitKey();
  }

  for(int i = -1; i < 2; ++i) {
    cv::imshow("main", image_augmentor.FlipImage(mat, i));
    string filename = save_dir + "fi" + (char)(i + '0') + extension;
    cv::imwrite(filename, image_augmentor.FlipImage(mat, i));
    cv::waitKey();
  }
  return 0;
}
