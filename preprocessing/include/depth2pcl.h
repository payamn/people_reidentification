#ifndef DEPTH2PCL_H

#include <iostream>
 
// Point cloud library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// OpenCV library
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
void depth2pcl(
    std::string depth_image_file, std::string rgb_image_file, std::string parameter_file, // input params
    PointCloudT::Ptr &cloud, cv::Mat &undistorted_rgb_image, cv::Mat &camera_matrix // output params
);

#endif