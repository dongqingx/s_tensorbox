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

  for(int i = 0; i < 119; ++i) {
    string file_num = "";
    file_num = file_num + (char)(i/100 + '0') + "" + (char)((i%100)/10 + '0') + "" + (char)((i%10) + '0');
    string jpg_name = "/home/donny/Projects/dataset/dlib_aligned2/train/" + file_num + ".jpg";
    cout << jpg_name << endl;
    // return 0;

    cv::Mat mat = cv::imread(jpg_name);
    cv::namedWindow("main");
    cv::imshow("main", mat);
    // cv::waitKey();
  
    string extension = file_num + ".png";
    string save_dir = "./a/";
    string ori_dir = "./b/";
  
    double start = clock();
  
    for(int i = 0; i < 10; ++i) {
      cv::imshow("main", image_augmentor.SimulateIllumination(mat, i % 4));
      string save_filename = save_dir + "sm" + (char)(i + '0') + extension;
      string ori_filename = ori_dir + "sm" + (char)(i + '0') + extension;
      cv::imwrite(ori_filename, mat);
      cv::imwrite(save_filename, image_augmentor.SimulateIllumination(mat, i % 4));
      // cv::waitKey();
      // image_augmentor.SimulateIllumination(mat, i % 4);
    }
    double end = clock();
    //cout << "Use time: " << end - start << endl;
    // string extension = ".png";
    // string save_dir = "./data/";
    for(int i = 0; i < 20; ++i) {
      cv::imshow("main", image_augmentor.ColorJittering(mat, false));
      string filename = save_dir + "cj" + (char)(i + '0') + extension;
      string ori_filename = ori_dir + "cj" + (char)(i + '0') + extension;
      cv::imwrite(ori_filename, mat);
      cv::imwrite(filename, image_augmentor.ColorJittering(mat, false));
      // cv::waitKey();
    }
    for(int i = 0; i < 2; ++i) {
      cv::imshow("main", image_augmentor.AddPepperSaltNoisy(mat, false));
      string filename1 = save_dir + "ps" + (char)(i + '0') + extension;
      string filename2 = save_dir + "gs" + (char)(i + '0') + extension;
      string ori_filename1 = ori_dir + "ps" + (char)(i + '0') + extension;
      string ori_filename2 = ori_dir + "gs" + (char)(i + '0') + extension;
      cv::imwrite(ori_filename1, mat);
      cv::imwrite(filename1, image_augmentor.AddPepperSaltNoisy(mat, false));
      // cv::waitKey();
      cv::imshow("main", image_augmentor.AddGaussianNoisy(mat, false));
      cv::imwrite(ori_filename2, mat);
      cv::imwrite(filename2, image_augmentor.AddGaussianNoisy(mat, false));
      // cv::waitKey();
    }

    for(int i =0; i < 2; ++i) {
      cv::imshow("main", image_augmentor.ScaleUpDown(mat, 0.30, 0.86, 5, 1));
      string filename = save_dir + "ud" + (char)(i + '0') + extension;
      string ori_filename = ori_dir + "ud" + (char)(i + '0') + extension;
      cv::imwrite(ori_filename, mat);
      cv::imwrite(filename, image_augmentor.ScaleUpDown(mat, 0.30, 0.86, 5, 1));
      // cv::waitKey();
    }


  }
/*
  for(int i = 0; i < 10; ++i) {
    cv::imshow("main", image_augmentor.RotateImage(mat, true));
    string filename = save_dir + "rt" + (char)(i + '0') + extension;
    string ori_filename = ori_dir + "rt" + (char)(i + '0') + extension;
    cv::imwrite(ori_filename, mat); 
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
*/
  return 0;
}
