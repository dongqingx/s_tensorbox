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


class ImageAugmentor {
  private:
    bool is_color;

  public:
    ImageAugmentor(bool is_color_ = true):is_color(is_color_){}

    // Simulate the illumination from different directions.
    cv::Mat SimulateIllumination(cv::Mat image, int direction, bool is_large_scale = false);

    // Color jittering with contrast & brightness.
    cv::Mat ColorJittering(cv::Mat image, bool is_large_scale = false);

    // Add gaussian noisy to images.
    cv::Mat AddNoisy(cv::Mat image, bool is_large_scale = false);

    // Scaling up & down.
    cv::Mat ScaleUpDown(cv::Mat image, float shrink_ratio_min, float shrink_ratio_max, 
                        int shrink_method, int enlarge_method);

    // Rotate the image.
    cv::Mat RotateImage(cv::Mat image, bool is_large_scale = false);

    // Random image augmentation.
    std::vector<cv::Mat> RandomAugment(cv::Mat image, float shrink_ratio_min, float shrink_ratio_max, 
                        int shrink_method, int enlarge_method, std::vector<int> exclude_methods, 
                        int rand_times = 1, bool is_random = false, bool is_large_scale = false);
};
