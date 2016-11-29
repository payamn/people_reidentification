#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>    
#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>

#include "depth2pcl.h"

#define PERSON_HALF_WIDTH 0.5
#define MIN_HEIGHT 1.3
#define MAX_HEIGHT 2.3
#define IMAGE_WIDTH 400
#define IMAGE_HEIGHT 400

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

enum { COLS = 640, ROWS = 480 };

int print_help()
{
  cout << "*******************************************************" << std::endl;
  cout << "Body detection:" << std::endl;
  cout << "   --help        <show_this_help>" << std::endl;
  cout << "   --depth       <path to depth image>" << std::endl;
  cout << "   --rgb         <path to rgb image>" << std::endl;
  cout << "   --camera      <path to camera info>" << std::endl;
  cout << "   --svm         <path to svm file>" << std::endl;
  cout << "   --pcd         <path to output pcd>" << std::endl;
  cout << "   --person-pcd  <path to output person pcd>" << std::endl;
  cout << "   --image       <path to output 2D image>" << std::endl;
  cout << "*******************************************************" << std::endl;
  return 0;
}

int main (int argc, char** argv)
{
   if(pcl::console::find_switch (argc, argv, "--help") || pcl::console::find_switch (argc, argv, "-h")) {
      return print_help();
   }

   // Algorithm parameters:
   std::string depth_image_path;
   std::string rgb_image_path;
   std::string svm_path;
   std::string camera_info_path;
   std::string pcd_path;
   std::string person_pcd_path;
   std::string image_path;
   // Read if some parameters are passed from command line:
   pcl::console::parse_argument (argc, argv, "--depth", depth_image_path);
   pcl::console::parse_argument (argc, argv, "--rgb", rgb_image_path);
   pcl::console::parse_argument (argc, argv, "--camera", camera_info_path);
   pcl::console::parse_argument (argc, argv, "--svm", svm_path);
   pcl::console::parse_argument (argc, argv, "--pcd", pcd_path);
   pcl::console::parse_argument (argc, argv, "--person-pcd", person_pcd_path);
   pcl::console::parse_argument (argc, argv, "--image", image_path);

   float min_confidence = -1.5;
   float voxel_size = 0.06;
   Eigen::Matrix3f rgb_intrinsics_matrix;
   
 
   PointCloudT::Ptr cloud (new PointCloudT);
   cv::Mat undistorted_rgb_image;
   cv::Mat camera_matrix;
   
   depth2pcl(
     depth_image_path,
     rgb_image_path,
     camera_info_path,
     cloud,
     undistorted_rgb_image,
     camera_matrix
   );

   // Save Cloud
   pcl::io::savePCDFileASCII (pcd_path, *cloud);
 
   rgb_intrinsics_matrix << 
     camera_matrix.at<float>(0, 0), camera_matrix.at<float>(0, 1), camera_matrix.at<float>(0, 2),
     camera_matrix.at<float>(1, 0), camera_matrix.at<float>(1, 1), camera_matrix.at<float>(1, 2),
     camera_matrix.at<float>(2, 0), camera_matrix.at<float>(2, 1), camera_matrix.at<float>(2, 2); // Kinect RGB camera intrinsics  
   
   // Ground plane estimation:
   Eigen::VectorXf ground_coeffs;
   ground_coeffs.resize(4);
//   ground_coeffs << 0.1349, -0.00246158, -0.990856, -1.22231;
   ground_coeffs << 0.0069949, 0.990188, 0.139569, -1.22829;
 
   // Create classifier for people detection:  
   pcl::people::PersonClassifier<pcl::RGB> person_classifier;
   person_classifier.loadSVMFromFile(svm_path);   // load trained SVM
 
   // People detection app initialization:
   pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
   people_detector.setVoxelSize(voxel_size);                        // set the voxel size
   people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
   people_detector.setClassifier(person_classifier);                // set person classifier
   people_detector.setHeightLimits(MIN_HEIGHT, MAX_HEIGHT);         // set person classifier
 //  people_detector.setSensorPortraitOrientation(true);             // set sensor orientation to vertical
 
   // Perform people detection on the new cloud:
   std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
   people_detector.setInputCloud(cloud);
   people_detector.setGround(ground_coeffs);                    // set floor coefficients
   people_detector.compute(clusters);                           // perform people detection
 
   ground_coeffs = people_detector.getGround();                 // get updated floor coefficients
 
   PointCloudT::Ptr no_ground_cloud (new PointCloudT);
   no_ground_cloud = people_detector.getNoGroundCloud();
 
   unsigned int k = 0;
   PointCloudT::Ptr cluster_cloud (new PointCloudT);
   PointCloudT::Ptr transformed_cloud (new PointCloudT);
   cv::Mat transformed_image( undistorted_rgb_image.rows, undistorted_rgb_image.cols, CV_8UC3, cv::Scalar::all(0) );
   for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
   {
     if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
     {
       Eigen::Vector3f top = it->getTop();
       Eigen::Vector3f bottom = it->getBottom();
 
       Eigen::Vector4f minPoints;
       minPoints[0] = top[0]-PERSON_HALF_WIDTH;
       minPoints[1] = top[1];
       minPoints[2] = top[2]-PERSON_HALF_WIDTH;
       minPoints[3] = 1;
 
       Eigen::Vector4f maxPoints;
       maxPoints[0] = bottom[0]+PERSON_HALF_WIDTH;
       maxPoints[1] = bottom[1];
       maxPoints[2] = bottom[2]+PERSON_HALF_WIDTH;
       maxPoints[3] = 1;
 
       pcl::CropBox<PointT> cropFilter;
       cropFilter.setInputCloud(cloud);
       cropFilter.setMin(minPoints);
       cropFilter.setMax(maxPoints);
       cropFilter.filter(*cluster_cloud);
 
       Eigen::Affine3f transform_origin = Eigen::Affine3f::Identity();
       transform_origin.translation() << -minPoints[0], -minPoints[1], -minPoints[2];
       pcl::transformPointCloud(*cluster_cloud, *transformed_cloud, transform_origin);
 
       // create new image with (r, g, b) = (x, y, z)
       float min_z = INFINITY;
       float min_x = INFINITY;
       float min_y = INFINITY;
       float max_x = -INFINITY;
       float max_y = -INFINITY;
       float max_z = -INFINITY;
       for (PointCloudT::iterator b1 = transformed_cloud->points.begin(); b1 < transformed_cloud->points.end(); b1++)
       {
         if (b1->x < min_x) {
            min_x = b1->x;
         }
         if (b1->x > max_x) {
            max_x = b1->x;
         }
         if (b1->y < min_y) {
            min_y = b1->y;
         }
         if (b1->y > max_y) {
            max_y = b1->y;
         }
         if (b1->z < min_z) {
            min_z = b1->z;
         }
         if (b1->z > max_z) {
            max_z = b1->z;
         }
       }
 
       int min_i = 1000;
       int min_j = 1000;
       int max_i = 0;
       int max_j = 0;
       for (PointCloudT::iterator b1 = cluster_cloud->points.begin(); b1 < cluster_cloud->points.end(); b1++)
       {
         // translated so that bottom is always origin
         float translated_x = b1->x - minPoints[0];
         float translated_y = b1->y - minPoints[1];
         float translated_z = b1->z - minPoints[2];

         // map the 3D coords back to 2D image coords  
         int i = (b1->x * camera_matrix.at<float>(0, 0) / b1->z) + camera_matrix.at<float>(0, 2);
         int j = (b1->y * camera_matrix.at<float>(1, 1) / b1->z) + camera_matrix.at<float>(1, 2);
 
         if (i > max_i) {
           max_i = i;
         }
         if (j > max_j) {
           max_j = j;
         }
         if (i < min_i) {
           min_i = i;
         }
         if (j < min_j) {
           min_j = j;
         }
         transformed_image.at<cv::Vec3b>(j, i)[0] = (translated_x - min_x) / (2 * PERSON_HALF_WIDTH) * 255;
         transformed_image.at<cv::Vec3b>(j, i)[1] = (translated_y - min_y) / (MAX_HEIGHT) * 255;
         transformed_image.at<cv::Vec3b>(j, i)[2] = (translated_z - min_z) / (2 * PERSON_HALF_WIDTH) * 255;
       }
 
       cv::Mat cropped_image(transformed_image, cv::Rect(min_i, min_j, max_i-min_i, max_j-min_j));
       cv::Mat resized_image;
       cv::resize(cropped_image, resized_image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
       cv::imwrite(image_path, resized_image);

       // Voxelization
       PointCloudT::Ptr voxel_filtered (new PointCloudT ());
       // Create the filtering object
       pcl::VoxelGrid<PointT> sor;
       sor.setInputCloud (transformed_cloud);
       sor.setLeafSize (0.025f, 0.06f, 0.025f);
       sor.filter (*voxel_filtered);
       
       // rotate the point cloud
       Eigen::Matrix4f transform;
       transform << 
         0,  0, 1, 0,
         -1, 0, 0, 0,
         0, -1, 0, 0,
         0,  0, 0, 1;
       PointCloudT::Ptr rotated_voxel_filtered(new PointCloudT());
       pcl::transformPointCloud(*voxel_filtered, *rotated_voxel_filtered, transform);
       pcl::io::savePCDFileASCII(person_pcd_path, *rotated_voxel_filtered);
  
 //      pcl::PCDWriter writer;
 //         writer.write ("table_scene_lms400_downsampled.pcd", *voxel_filtered, 
 //         Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);
 

 /*
       // draw theoretical person bounding box in the PCL viewer:
       pcl::PointIndices clusterIndices = it->getIndices();    // cluster indices
       std::vector<int> indices = clusterIndices.indices;
       for(unsigned int i = 0; i < indices.size(); i++)        // fill cluster cloud
       {
         PointT* p = &no_ground_cloud->points[indices[i]];
         cluster_cloud->push_back(*p);
       }
 */
       k++;
     }
   }
   std::cout << k << " people found" << std::endl;
 
   return 0;
}
