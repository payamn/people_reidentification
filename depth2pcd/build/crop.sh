K=02
S=01
A=01
R=01
prefix=Kin${K}/S${S}/A${A}/R${R}/kin_k${K}_s${S}_a${A}_r${R}
camera_path="~/workspace/people_reidentification/dataset/demo/BerkeleyMHAD/Calibration/camcfg_k02.yml"
for i in {00001..00173}
do
  color_path="/home/faraz/workspace/people_reidentification/dataset/demo/BerkeleyMHAD/Kinect/${prefix}_color_${i}.ppm"
  depth_path="/home/faraz/workspace/people_reidentification/dataset/demo/BerkeleyMHAD/Kinect/${prefix}_depth_${i}.pgm"
  pcd_path="/home/faraz/workspace/people_reidentification/dataset/demo/BerkeleyMHAD/Kinect/${prefix}_pcd_${i}.pcd"
  ./depth2pcd --camera ${camera_path} --depth ${depth_path} --rgb ${color_path} --output ${pcd_path}
done
