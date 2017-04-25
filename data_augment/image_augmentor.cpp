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


cv::Mat ImageAugmentor::SimulateIllumination(cv::Mat image, int direction, bool is_large_scale) {
  double bright_factor = 0.0;
  if(!is_large_scale)
  {
    bright_factor = (rand() % 10 + 5.0) / 10.0;
  }
  else
  {
    bright_factor = (rand() % 10 + 5.0) / 10.0;
  }
  // cout << "bright_factor: " << bright_factor << endl;
  cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());
  int width = image.cols;
  int height = image.rows;
  // cout << "image size: " << image.size() << endl;
  // cv::Mat new_image(width, height, CV_8UC3, cv::Scalar(0,0,0));

  double alpha = 0.0;

  switch(direction){
    case 0:
      for(int i = 0; i < width; ++i)
      {
        alpha = ((width - 2.0 * i) / (width * 1.0)) * bright_factor;
        // cout << "alpha: " << alpha << endl;
        for(int j = 0; j < height; ++j)
        {
          for(int k = 0; k < 3; ++k)
          {
            int pixel = MIN((1.0 + alpha) * image.at<cv::Vec3b>(j, i)[k], 255);
            new_image.at<cv::Vec3b>(j, i)[k] = cv::saturate_cast<uchar>(pixel);
          }
        }
      }
      break;
    case 1:
      for(int i = 0; i < width; ++i)
      {
        alpha = ((width - 2.0 * i) / (width * 1.0)) * bright_factor;
        // cout << "alpha: " << alpha << endl;
        for(int j = 0; j < height; ++j)
        {
          for(int k = 0; k < 3; ++k)
          {
            int pixel = MIN((1.0 - alpha) * image.at<cv::Vec3b>(j, i)[k], 255);
            new_image.at<cv::Vec3b>(j, i)[k] = cv::saturate_cast<uchar>(pixel);
          }
        }
      } 
      break;

    case 2:
      for(int i = 0; i < height; ++i)
      {
        alpha = ((height - 2.0 * i) / (height * 1.0)) * bright_factor;
        // cout << "alpha: " << alpha << endl;
        for(int j = 0; j < width; ++j)
        {
          for(int k = 0; k < 3; ++k)
          {
            int pixel = MIN((1.0 + alpha) * image.at<cv::Vec3b>(i, j)[k], 255);
            new_image.at<cv::Vec3b>(i, j)[k] = cv::saturate_cast<uchar>(pixel);
          }
        }
      }
      break;

    case 3:
      for(int i = 0; i < height; ++i)
      {
        alpha = ((height - 2.0 * i) / (height * 1.0)) * bright_factor;
        // cout << "alpha: " << alpha << endl;
        for(int j = 0; j < width; ++j)
        {
          for(int k = 0; k < 3; ++k)
          {
            int pixel = MIN((1.0 - alpha) * image.at<cv::Vec3b>(i, j)[k], 255);
            new_image.at<cv::Vec3b>(i, j)[k] = cv::saturate_cast<uchar>(pixel);
          }
        }
      }
      break;
  }
  return new_image;
}

cv::Mat ImageAugmentor::ColorJittering(cv::Mat image, bool is_large_scale) {
  cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());
  double contrast_fac = 0.0;
  int bright_fac = 0;
  if(!is_large_scale)
  {
    contrast_fac = (rand() / double(RAND_MAX)) * 1.0 + 0.75;
    bright_fac = rand() % 150 - 85;
  }
  else
  {
    contrast_fac = (rand() / double(RAND_MAX)) * 2.0 + 1.0;
    bright_fac = rand() % 100;
  }
  // cout << "contrast_fac: " << contrast_fac << "bright_fac: " << bright_fac << endl;
  for(int y = 0; y < image.rows; y++)
  {
    for(int x = 0; x < image.cols; x++)
    {
      for(int c = 0; c < 3; c++)
      {
        int pixel = contrast_fac * (image.at<cv::Vec3b>(y,x)[c]) + bright_fac;
        new_image.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(pixel);
      }
    }
  }
  return new_image;
}

double generateGaussianNoise(double mu, double sigma)  
{  
    const double epsilon = std::numeric_limits<double>::min(); 
    static double z0, z1;  
    static bool flag = false;  
    flag = !flag;  
    if (!flag)  
        return z1*sigma + mu;  
    double u1, u2;  
      
    do  
    {  
        u1 = rand()*(1.0 / RAND_MAX);  
        u2 = rand()*(1.0 / RAND_MAX);  
    } while (u1 <= epsilon);  
    z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI * u2);  
    z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI * u2);  
    return z1*sigma + mu;  
}
  
cv::Mat ImageAugmentor::AddGaussianNoisy(cv::Mat srcImage, bool is_large_scale) {
    cv::Mat resultImage = srcImage.clone();
    int channels = resultImage.channels();
    int nRows = resultImage.rows;
    int nCols = resultImage.cols*channels;
    if (resultImage.isContinuous())
    {  
        nCols *= nRows;
        nRows = 1;
    }
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            int val = resultImage.ptr<uchar>(i)[j] + generateGaussianNoise(2, 0.8) * 32;
            if (val < 0) 
                val = 0;
            if (val > 255) 
                val = 255;
            resultImage.ptr<uchar>(i)[j] = (uchar)val;
        }
    }
    return resultImage;
}

cv::Mat ImageAugmentor::AddPepperSaltNoisy(cv::Mat image, bool is_large_scale) {
  cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());
  int pixel_cnt = image.rows * image.cols;
  int skip_count = 0;
  int rand_skip = rand() % 50;
  int white_or_black = rand() % 2;
  bool is_checkpoint = false;

  for(int y = 0; y < image.rows; y++)
  {
    for(int x = 0; x < image.cols; x++)
    {
      skip_count ++;
      if(skip_count > rand_skip) {
        is_checkpoint = true;
        rand_skip = rand() % 50;
        white_or_black = rand() % 2;
        skip_count = 0;
      }
      for(int c = 0; c < 3; c++)
      {
        int pixel = 0;
        if(is_checkpoint) {
            pixel = white_or_black == 0? 255:0;
        }
        else {
            pixel = image.at<cv::Vec3b>(y,x)[c];
        }
        new_image.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(pixel);
      }
      is_checkpoint = false;
    }
  }
 
  return new_image;
}

cv::Mat ImageAugmentor::ScaleUpDown(cv::Mat image, float shrink_ratio_min, 
                                    float shrink_ratio_max, int shrink_method, int enlarge_method) {
  cv::Mat cv_img = image.clone();
  cv::Mat shrink_img;

  int rand_times = rand() % 5;

  for(int i = 0; i < rand_times; ++i) {
    
      float shrink_ratio = (rand() / (RAND_MAX + 1.0)) * (shrink_ratio_max - shrink_ratio_min) + shrink_ratio_min;
    
      int  origin_width = cv_img.cols, origin_height = cv_img.rows;
      int shrink_width = round(shrink_ratio * origin_width);
      int shrink_height = round(shrink_ratio * origin_height);
      int shrink_interpolation = CV_INTER_LINEAR, enlarge_interpolation = CV_INTER_LINEAR;
    
      switch(shrink_method) {
        case 0:
          shrink_interpolation = CV_INTER_NN;
          break;
        case 1:
          shrink_interpolation = CV_INTER_LINEAR;
          break;
        case 2:
          shrink_interpolation = CV_INTER_CUBIC;
          break;
        case 3:
          shrink_interpolation = CV_INTER_AREA;
          break;
        case 4:
          shrink_interpolation = CV_INTER_LANCZOS4;
          break;
        case 5:
          switch(rand() % 5){
            case 0 : {shrink_interpolation = CV_INTER_NN; break;}
            case 1 : {shrink_interpolation = CV_INTER_LINEAR ; break;}
            case 2 : {shrink_interpolation = CV_INTER_CUBIC  ; break;}
            case 3 : {shrink_interpolation = CV_INTER_AREA   ; break;}
            case 4 : {shrink_interpolation = CV_INTER_LANCZOS4; break;}
            default: break;
          }
      }
    
      cv::resize(cv_img, shrink_img, cv::Size(shrink_width, shrink_height), 0.0, 0.0, shrink_interpolation);
    
      switch(enlarge_method) {
        case 0:
          enlarge_interpolation = CV_INTER_NN;
          break;
        case 1:
          enlarge_interpolation = CV_INTER_LINEAR;
          break;
        case 2:
          enlarge_interpolation = CV_INTER_CUBIC;
          break;
        case 3:
          enlarge_interpolation = CV_INTER_AREA;
          break;
        case 4:
          enlarge_interpolation = CV_INTER_LANCZOS4;
          break;
        case 5:
          switch(rand() % 5){
            case 0 : {enlarge_interpolation = CV_INTER_NN; break;}
            case 1 : {enlarge_interpolation = CV_INTER_LINEAR ; break;}
            case 2 : {enlarge_interpolation = CV_INTER_CUBIC  ; break;}
            case 3 : {enlarge_interpolation = CV_INTER_AREA   ; break;}
            case 4 : {enlarge_interpolation = CV_INTER_LANCZOS4; break;}
            default: break;
          }
      }
    
      cv::resize(shrink_img, cv_img, cv::Size(origin_width, origin_height), 0.0, 0.0, enlarge_interpolation);
  }
  return cv_img;
}

cv::Mat ImageAugmentor::RotateImage(cv::Mat image, bool is_large_scale) {
  cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());
  double rand_angle = 0.0;
  double shrink_scale = 1.0;

  if(!is_large_scale)
  {
    rand_angle = (rand() / double(RAND_MAX)) * 30.0 - 15.0;
  }
  else
  {
    rand_angle = (rand() / double(RAND_MAX)) * 60.0 - 30.0;
  }
  cv::Point2f center = cv::Point2f(image.cols / 2, image.rows / 2);

  cv::Mat rotate_mat = cv::getRotationMatrix2D(center, rand_angle, shrink_scale);
  cv::warpAffine(image, new_image, rotate_mat, image.size());

  return new_image;
}

cv::Mat ImageAugmentor::FlipImage(cv::Mat image, int flip_method) {
  cv::Mat new_image;
  cv::flip(image, new_image, flip_method);
  return new_image;
}

std::vector<cv::Mat> ImageAugmentor::RandomAugment(cv::Mat image, float shrink_ratio_min, 
    float shrink_ratio_max, int shrink_method, int enlarge_method, std::vector<int> exclude_methods, 
    int rand_times, bool is_random, bool is_large_scale) {
  std::vector<cv::Mat> result_mats;
  // cv::Mat clone() the element of vector.

  if(is_random)
  {
      cv::Mat new_image;
      int rand_method = rand() % 5;
      int rand_direction = 0;
      std::vector<int>::iterator iter = find(exclude_methods.begin(),exclude_methods.end(), rand_method);
      if(iter != exclude_methods.end())
      {
        rand_method = 4;
      }
      switch(rand_method) {
        case 0:
          rand_direction = rand() % 4;
          new_image = SimulateIllumination(image, rand_direction, is_large_scale);
          result_mats.push_back(new_image);
          break;
        case 1:
          new_image = ColorJittering(image, is_large_scale);
          result_mats.push_back(new_image);
          break;
        case 2:
          new_image = AddPepperSaltNoisy(image, is_large_scale);
          result_mats.push_back(new_image);
          break;
        case 3:
          new_image = ScaleUpDown(image, shrink_ratio_min, 
                                  shrink_ratio_max, shrink_method, enlarge_method);
          result_mats.push_back(new_image);
          break;
        case 4:
          // Do nothing.
          break;

        default:
          break;
      }
  }
  else
  {
    cv::Mat temp_mat;
    for(int time = 0; time < rand_times; ++time)
    {
      for(int i = 0; i < 4; ++i)
      {
        temp_mat = SimulateIllumination(image, i, is_large_scale);
        result_mats.push_back(temp_mat);
      }
      temp_mat = ColorJittering(image, is_large_scale);
      result_mats.push_back(temp_mat);

      temp_mat = AddPepperSaltNoisy(image, is_large_scale);
      result_mats.push_back(temp_mat);

      temp_mat = ScaleUpDown(image, shrink_ratio_min, 
                             shrink_ratio_max, shrink_method, enlarge_method);
      result_mats.push_back(temp_mat);
    }
  }
  return result_mats;
}
