#include <iostream>
 
// Point cloud library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
 
// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// boost
#include <boost/thread/thread.hpp>

int print_help()
{
  cout << "*******************************************************" << std::endl;
  cout << "RGB-D to PCD:" << std::endl;
  cout << "   --help    <show_this_help>" << std::endl;
  cout << "   --depth   <path to depth file>" << std::endl;
  cout << "   --rgb     <path to rgb file>" << std::endl;
  cout << "   --camera  <path to camera info file>" << std::endl;
  cout << "   --output  <output name>" << std::endl;
  cout << "*******************************************************" << std::endl;
  return 0;
}

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
	
int main( int argc, char *argv[] ) {
   if(pcl::console::find_switch (argc, argv, "--help") || pcl::console::find_switch (argc, argv, "-h")) {
      return print_help();
   }

   // Algorithm parameters:
   std::string depth_image_path;
   std::string rgb_image_path;
   std::string camera_info_path;
   std::string output_name;

   // Read if some parameters are passed from command line:
   pcl::console::parse_argument (argc, argv, "--depth", depth_image_path);
   pcl::console::parse_argument (argc, argv, "--rgb", rgb_image_path);
   pcl::console::parse_argument (argc, argv, "--camera", camera_info_path);
   pcl::console::parse_argument (argc, argv, "--output", output_name);

   Eigen::Matrix3f rgb_intrinsics_matrix;
   rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

   cv::FileStorage fs2(camera_info_path, cv::FileStorage::READ);

   cv::Mat cameraMatrix, distCoeffs;
   fs2["Camera_1"]["K"] >> cameraMatrix;
   fs2["Camera_1"]["Dist"] >> distCoeffs;

   std::cout << distCoeffs << std::endl;

   const float depth_factor = 1000; // mapping from png depth value to metric scale

   // TODO load these constant from the cameraMatrix
   const float fx_d = 531.49230957f;
   const float fy_d = 532.39190674f;
   const float px_d = 314.63775635f;
   const float py_d = 252.53335571f;

   pcl::PointCloud<pcl::PointXYZ> cloud;
   pcl::PointCloud<pcl::PointXYZRGB> rgb_cloud;
    
   cv::Mat depth_image = cv::imread(
      depth_image_path,
      CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR 
   );

   cv::Mat rgb_image = cv::imread(
      rgb_image_path,
      CV_LOAD_IMAGE_COLOR
   );

   depth_image.convertTo( depth_image, CV_32F ); // convert the image data to float type
   // rgb_image.convertTo( rgb_image, CV_32F); // convert the image data to float type

   if(!depth_image.data){
      std::cerr << "No depth data!!!" << std::endl;
      exit(EXIT_FAILURE);
   }

   if(!rgb_image.data){
      std::cerr << "No RGB data!!!" << std::endl;
      exit(EXIT_FAILURE);
   }

   std::cout << distCoeffs;
   // undistort the rgb image
   {
       cv::Mat temp_image;
       cv::undistort(rgb_image, temp_image, cameraMatrix, distCoeffs);
       temp_image.copyTo(rgb_image);
   }


   cloud.width = depth_image.cols; //Dimensions must be initialized to use 2-D indexing
   cloud.height = depth_image.rows;
   cloud.resize(cloud.width*cloud.height);

   rgb_cloud.width = depth_image.cols; //Dimensions must be initialized to use 2-D indexing
   rgb_cloud.height = depth_image.rows;
   rgb_cloud.resize(rgb_cloud.width*rgb_cloud.height);

   uint8_t* rgb_data = (uint8_t*)rgb_image.data;

   for(int v=0; v < depth_image.rows; v++) {
      //2-D indexing
      for(int u = 0; u < depth_image.cols; u++) {
         float depth  = depth_image.at<float>(v,u) / depth_factor;
          
         float z = depth;
         float x = depth*(u-px_d)/fx_d;
         float y = depth*(v-py_d)/fy_d;
           
         cloud(u,v).x = x;
         cloud(u,v).y = y;
         cloud(u,v).z = z;

         rgb_cloud(u,v).x = x;
         rgb_cloud(u,v).y = y;
         rgb_cloud(u,v).z = z;

         rgb_cloud(u,v).b = rgb_data[v * rgb_image.cols * 3 + u * 3 + 0];
         rgb_cloud(u,v).g = rgb_data[v * rgb_image.cols * 3 + u * 3 + 1];
         rgb_cloud(u,v).r = rgb_data[v * rgb_image.cols * 3 + u * 3 + 2];
      } 
   }

   pcl::io::savePCDFileASCII (output_name, rgb_cloud);
    
   // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr( &cloud );
   // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = simpleVis( cloud_ptr );

   // pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_ptr( &rgb_cloud );
   // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = rgbVis( cloud_ptr );

   // while (!viewer->wasStopped ()) {
   //     viewer->spinOnce (100);
   //     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
   // }

   // pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
   // viewer.showCloud (cloud);
}

