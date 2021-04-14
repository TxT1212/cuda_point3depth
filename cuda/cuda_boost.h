#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// d_points_W_XYZ: point cloud xyz in world coordination
// d_points_C_XYZ: point cloud xyz in camera coordination
// d_points_u, d_points_v: uv of points cast into pixel coordination
// d_points_valid: whether point cast into pixel is in the frame
// W2L: T from world coordination to camera coordination
// N: number of points
// width, height, fx, fy, px, py: Camera parameters  
void getpointsparam(Eigen::Vector3d* d_points_W_XYZ, Eigen::Vector3d* d_points_C_XYZ, double* d_points_u, double* d_points_v, bool* d_points_valid, Eigen::Matrix4d W2L, int N, int width, int height, double fx, double fy, double px, double py);

void getimagedata(Eigen::Vector3d* d_faces, unsigned char* d_depth, bool* d_points_valid, double* d_points_u, double* d_points_v, Eigen::Vector3d* d_points_C_XYZ, int M, int width, int height, double fx, double fy, double px, double py, int* face_result, double* u_mat, double* v_mat);
