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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

enum { COLS = 640, ROWS = 480 };

int print_help()
{
  cout << "*******************************************************" << std::endl;
  cout << "Body detection:" << std::endl;
  cout << "   --help    <show_this_help>" << std::endl;
  cout << "   --pcd     <path to pcd file>" << std::endl;
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
   std::string output_name;
   // Read if some parameters are passed from command line:
   pcl::console::parse_argument (argc, argv, "--pcd", pcd_path);
   pcl::console::parse_argument (argc, argv, "--output", output_name);

   PointCloudT::Ptr cloud (new PointCloudT);
   // Display pointcloud:
   if (pcl::io::loadPCDFile<PointT> (pcd_path, *cloud) == -1) //* load the file
   {
      PCL_ERROR ("Couldn't read pcd file \n");
      return (-1);
   }

   float min_z = INFINITY;
   float min_x = INFINITY;
   float min_y = INFINITY;
   float max_x = -INFINITY;
   float max_y = -INFINITY;
   float max_z = -INFINITY;

   pcl::PointCloud<PointT>::iterator b1;
   for (b1 = cloud->points.begin(); b1 < cloud->points.end(); b1++)
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

   cv::Mat image(400, 400, CV_8UC1, cv::Scalar::all(0));
   for (b1 = cloud->points.begin(); b1 < cloud->points.end(); b1++)
   {
      int i = (b1->x - min_x) / (max_x - min_x) * 399;
      int j = (b1->y - min_y) / (max_y - min_y) * 399;
      image.at<uchar> (j, i) = (b1->z - min_z)  / (max_z - min_z) * 255;
   }

   cv::imshow("image", image);
   cv::waitKey(0);

   return 0;
}
