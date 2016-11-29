#include <iostream>
 
// Point cloud library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
 
// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// boost
#include <boost/thread/thread.hpp>

#include "depth2pcl.h"

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
   // --------------------------------------------
   // -----Open 3D viewer and add point cloud-----
   // --------------------------------------------
   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
   viewer->setBackgroundColor (0, 0, 0);
   viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
   viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
   viewer->addCoordinateSystem (1.0);
   viewer->initCameraParameters ();
   return (viewer);
 }

 boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


void depth2pcl(
  std::string depth_image_file, std::string rgb_image_file, std::string parameter_file, // input params
  PointCloudT::Ptr &cloud, cv::Mat &undistorted_rgb_image, cv::Mat &camera_matrix // output params
) {

  // read the files
  cv::FileStorage fs2(parameter_file, cv::FileStorage::READ);

  // cv::Mat camera_matrix;
  cv::Mat distortion_coeffs;
    fs2["Camera_1"]["K"] >> camera_matrix;
    fs2["Camera_1"]["Dist"] >> distortion_coeffs;

  cv::Mat depth_image = cv::imread(
      depth_image_file,
      CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR 
  );
  depth_image.convertTo( depth_image, CV_32F ); // convert the image data to float type

  cv::Mat rgb_image = cv::imread(
    rgb_image_file, 
    CV_LOAD_IMAGE_COLOR
  );

  if(!depth_image.data) {
      std::cerr << "No depth data!!!" << std::endl;
      exit(EXIT_FAILURE);
  }

  if(!rgb_image.data) {
      std::cerr << "No RGB data!!!" << std::endl;
      exit(EXIT_FAILURE);
  }

  // undistort rgb_image
  cv::undistort(rgb_image, undistorted_rgb_image, camera_matrix, distortion_coeffs);
  undistorted_rgb_image.copyTo(rgb_image);
 
  uint8_t* rgb_data = (uint8_t*)rgb_image.data;

  PointCloudT rgb_cloud;

  rgb_cloud.width = depth_image.cols;
  rgb_cloud.height = depth_image.rows;
  rgb_cloud.resize( rgb_cloud.width * rgb_cloud.height );

  const float depth_factor = 1000; // mapping from png depth value to metric scale
  // TODO load these constant from the camera_matrix
  const float fx_d = camera_matrix.at<float>(0, 0); // 531.49230957f;
  const float fy_d = camera_matrix.at<float>(1, 1); // 532.39190674f;
  const float px_d = camera_matrix.at<float>(0, 2); // 314.63775635f;
  const float py_d = camera_matrix.at<float>(1, 2); // 252.53335571f;

  for(int v=0; v < depth_image.rows; v++) {
    //2-D indexing
    for(int u = 0; u < depth_image.cols; u++) {
      
      float depth  = depth_image.at<float>(v,u) / depth_factor;
      if (depth == 0) {
          continue;
      }

      float z = depth;
      float x = depth*(u-px_d)/fx_d;
      float y = depth*(v-py_d)/fy_d;

      rgb_cloud(u,v).x = x;
      rgb_cloud(u,v).y = y;
      rgb_cloud(u,v).z = z;
      rgb_cloud(u,v).b = rgb_data[v * rgb_image.cols * 3 + u * 3 + 0];
      rgb_cloud(u,v).g = rgb_data[v * rgb_image.cols * 3 + u * 3 + 1];
      rgb_cloud(u,v).r = rgb_data[v * rgb_image.cols * 3 + u * 3 + 2];
    } 
  }
  
  cloud.reset( new PointCloudT(rgb_cloud) );
}
	

/*int main( int argc, char *argv[] ) {
    PointCloudT::Ptr cloud(
      depth2pcl(
        "/home/rakesh/rakesh/cmpt726/BerkeleyMHAD/Kinect/Kin01/S01/A01/R01/kin_k01_s01_a01_r01_depth_00000.pgm",
        "/home/rakesh/rakesh/cmpt726/BerkeleyMHAD/Kinect/Kin01/S01/A01/R01/kin_k01_s01_a01_r01_color_00000.ppm",
        "/home/rakesh/rakesh/cmpt726/BerkeleyMHAD/Calibration/camcfg_k01.yml"
      )
    );

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = rgbVis( cloud );

    while (!viewer->wasStopped ()) {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    
    
}*/

