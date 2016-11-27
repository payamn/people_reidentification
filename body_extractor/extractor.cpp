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

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

enum { COLS = 640, ROWS = 480 };

int print_help()
{
  cout << "*******************************************************" << std::endl;
  cout << "Body detection:" << std::endl;
  cout << "   --help    <show_this_help>" << std::endl;
  cout << "   --pcd     <path to pcd file>" << std::endl;
  cout << "   --svm     <path to svm file>" << std::endl;
  cout << "   --output  <output name>" << std::endl;
  cout << "*******************************************************" << std::endl;
  return 0;
}


int main (int argc, char** argv)
{
   if(pcl::console::find_switch (argc, argv, "--help") || pcl::console::find_switch (argc, argv, "-h")) {
      return print_help();
   }

   // Algorithm parameters:
   std::string pcd_path;
   std::string svm_path;
   std::string output_name;
   // Read if some parameters are passed from command line:
   pcl::console::parse_argument (argc, argv, "--pcd", pcd_path);
   pcl::console::parse_argument (argc, argv, "--svm", svm_path);
   pcl::console::parse_argument (argc, argv, "--output", output_name);

   float min_confidence = -1.5;
   float min_height = 1.3;
   float max_height = 2.3;
   float body_size = 0.5;
   float voxel_size = 0.06;
   Eigen::Matrix3f rgb_intrinsics_matrix;
   rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

   // Read Kinect live stream:
   PointCloudT::Ptr cloud (new PointCloudT);
   // Display pointcloud:
   if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcd_path, *cloud) == -1) //* load the file
   {
      PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
      return (-1);
   }

   // Ground plane estimation:
   Eigen::VectorXf ground_coeffs;
   ground_coeffs.resize(4);
   ground_coeffs << -0.00285197, 0.989396, 0.145214, -1.24224;
//   ground_coeffs.resize(4);

   // Create classifier for people detection:  
   pcl::people::PersonClassifier<pcl::RGB> person_classifier;
   person_classifier.loadSVMFromFile(svm_path);   // load trained SVM

   // People detection app initialization:
   pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
   people_detector.setVoxelSize(voxel_size);                        // set the voxel size
   people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
   people_detector.setClassifier(person_classifier);                // set person classifier
   people_detector.setHeightLimits(min_height, max_height);         // set person classifier
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
   for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
   {
      if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
      {
         Eigen::Vector3f top = it->getTop();
         Eigen::Vector3f bottom = it->getBottom();

         Eigen::Vector4f minPoints;
         minPoints[0] = top[0]-body_size;
         minPoints[1] = top[1];
         minPoints[2] = top[2]-body_size;
         minPoints[3] = 1;

         Eigen::Vector4f maxPoints;
         maxPoints[0] = bottom[0]+body_size;
         maxPoints[1] = bottom[1];
         maxPoints[2] = bottom[2]+body_size;
         maxPoints[3] = 1;

         pcl::CropBox<PointT> cropFilter;
         cropFilter.setInputCloud(cloud);
         cropFilter.setMin(minPoints);
         cropFilter.setMax(maxPoints);
         cropFilter.filter(*cluster_cloud);

         Eigen::Affine3f transform_origin = Eigen::Affine3f::Identity();
         transform_origin.translation() << -minPoints[0], -minPoints[1], -minPoints[2];
         pcl::transformPointCloud(*cluster_cloud, *transformed_cloud, transform_origin);
         pcl::io::savePCDFileASCII (output_name, *transformed_cloud);


         /// Create the filtering object
         // PointCloudT::Ptr voxel_filtered (new PointCloudT ());
         // pcl::VoxelGrid<PointT> sor;
         // sor.setInputCloud (transformed_cloud);
         // sor.setLeafSize (0.01f, 0.01f, 0.01f);
         // sor.filter (*voxel_filtered);

         // pcl::PCDWriter writer;
         // writer.write ("table_scene_lms400_downsampled.pcd", *voxel_filtered, 
         // Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);

         // pcl::io::savePCDFileASCII (output_name, *voxel_filtered);

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
