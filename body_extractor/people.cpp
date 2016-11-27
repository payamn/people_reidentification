#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>    
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/compression/octree_pointcloud_compression.h>

// #include <pcl/octree/octree_pointcloud_density.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
// #include <pcl/octree/octree_pointcloud_pointvector.h>
// #include <pcl/octree/octree_pointcloud_changedetector.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

// PCL viewer //
pcl::visualization::PCLVisualizer viewer("PCL Viewer");

// Mutex: //
boost::mutex cloud_mutex;

enum { COLS = 640, ROWS = 480 };

struct callback_args{
  // structure used to pass arguments to the callback function
  PointCloudT::Ptr clicked_points_3d;
  pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};
 
void
pp_callback (const pcl::visualization::PointPickingEvent& event, void* args)
{
  struct callback_args* data = (struct callback_args *)args;
  if (event.getPointIndex () == -1)
    return;
  PointT current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);
  data->clicked_points_3d->points.push_back(current_point);
  // Draw clicked points in red:
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red (data->clicked_points_3d, 255, 0, 0);
  data->viewerPtr->removePointCloud("clicked_points");
  data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
  data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
  std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}

int main (int argc, char** argv)
{
  // Algorithm parameters:
  std::string svm_filename = "trainedLinearSVMForPeopleDetectionWithHOG.yaml";
  float min_confidence = -1.5;
  float min_height = 1.3;
  float max_height = 2.3;
  float voxel_size = 0.06;
  Eigen::Matrix3f rgb_intrinsics_matrix;
  rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

  // Read Kinect live stream:
  PointCloudT::Ptr cloud (new PointCloudT);
  // Display pointcloud:
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> ("test.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    return (-1);
  }
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Add point picking callback to viewer:
  struct callback_args cb_args;
  PointCloudT::Ptr clicked_points_3d (new PointCloudT);
  cb_args.clicked_points_3d = clicked_points_3d;
  cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(&viewer);
  viewer.registerPointPickingCallback (pp_callback, (void*)&cb_args);
  std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

  // Spin until 'Q' is pressed:
  viewer.spin();
  std::cout << "done." << std::endl;
  
  cloud_mutex.unlock ();    

  // Ground plane estimation:
  Eigen::VectorXf ground_coeffs;
  ground_coeffs.resize(4);
  std::vector<int> clicked_points_indices;
  for (unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
    clicked_points_indices.push_back(i);
  pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
  model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
  std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

  // Initialize new viewer:
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");          // viewer initialization
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Create classifier for people detection:  
  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM

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

  // Draw cloud and people bounding boxes in the viewer:
//  viewer.removeAllPointClouds();
//  viewer.removeAllShapes();
//  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
//  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
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
      minPoints[0] = top[0]-0.5;
      minPoints[1] = top[1];
      minPoints[2] = top[2]-0.5;
      minPoints[3] = 1;

      Eigen::Vector4f maxPoints;
      maxPoints[0] = bottom[0]+0.5;
      maxPoints[1] = bottom[1];
      maxPoints[2] = bottom[2]+0.5;
      maxPoints[3] = 1;

      std::cout << maxPoints << std::endl << " " << std::endl << minPoints << std::endl;

      pcl::CropBox<PointT> cropFilter;
      cropFilter.setInputCloud(cloud);
      cropFilter.setMin(minPoints);
      cropFilter.setMax(maxPoints);
      cropFilter.filter(*cluster_cloud);

      Eigen::Affine3f transform_origin = Eigen::Affine3f::Identity();
      transform_origin.translation() << -minPoints[0], -minPoints[1], -minPoints[2];
      pcl::transformPointCloud(*cluster_cloud, *transformed_cloud, transform_origin);

        // octree
//      float resolution = 50.0;
//      pcl::octree::OctreePointCloudVoxelCentroid<PointT> octree(resolution);
//      octree.setInputCloud(transformed_cloud);
//      octree.addPointsFromInputCloud();


      PointCloudT::Ptr voxel_filtered (new PointCloudT ());

      // Create the filtering object
      pcl::VoxelGrid<PointT> sor;
      sor.setInputCloud (transformed_cloud);
      // sor.setLeafSize (0.01f, 0.01f, 0.01f);
      sor.setLeafSize (0.025f, 0.06f, 0.025f);
      sor.filter (*voxel_filtered);

      std::cerr << "PointCloud after filtering: " << voxel_filtered->width * voxel_filtered->height 
       << " data points (" << pcl::getFieldsList (*voxel_filtered) << ").";

//      pcl::PCDWriter writer;
//         writer.write ("table_scene_lms400_downsampled.pcd", *voxel_filtered,
//         Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);

      // for a full list of profiles see: /io/include/pcl/compression/compression_profiles.h
      // pcl::io::compression_Profiles_e compressionProfile = pcl::io::MED_RES_ONLINE_COMPRESSION_WITH_COLOR;

      //1 cubic centimetre = 0.01
      const double pointResolution_arg = .06;
      const double octreeResolution_arg = .06;
      bool doVoxelGridDownDownSampling_arg = true;
      const unsigned int iFrameRate_arg = 30;
      bool doColorEncoding_arg = true;
      const unsigned char colorBitResolution_arg = 5;

      // instantiate point cloud compression for encoding and decoding
      pcl::io::OctreePointCloudCompression<PointT>* PointCloudEncoder = new pcl::io::OctreePointCloudCompression<PointT> (pcl::io::MANUAL_CONFIGURATION,
           true,
           pointResolution_arg,
           octreeResolution_arg,
           doVoxelGridDownDownSampling_arg,
           iFrameRate_arg,
           doColorEncoding_arg,
           colorBitResolution_arg
      );

      pcl::io::OctreePointCloudCompression<PointT>* PointCloudDecoder = new pcl::io::OctreePointCloudCompression<PointT> ();

      std::stringstream compressedData;
      // output pointcloud
      PointCloudT::Ptr cloudOut (new PointCloudT ());

      // compress point cloud
      PointCloudEncoder->encodePointCloud (transformed_cloud, compressedData);

      // decompress point cloud
      PointCloudDecoder->decodePointCloud (compressedData, cloudOut);

      pcl::io::savePCDFileASCII ("person_octree.pcd", *cloudOut);
      pcl::io::savePCDFileASCII ("person_voxel.pcd", *voxel_filtered);

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

      it->drawTBoundingBox(viewer, k);

      viewer.removeAllPointClouds();
      viewer.removeAllShapes();
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(voxel_filtered);
      viewer.addPointCloud<PointT> (voxel_filtered, rgb, "cluster_cloud");
      viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cluster_cloud");
      viewer.spin();

      k++;
    }
  }
  std::cout << k << " people found" << std::endl;
  viewer.spin();

  return 0;
}
