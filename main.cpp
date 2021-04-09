#if (defined __GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ >= 7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

#include <iostream>
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h> //pcd 读写类相关的头文件。
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/io.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h> //PCL中支持的点类型头文件。
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/range_image/range_image.h>              //关于深度图像的头文件
#include <pcl/visualization/range_image_visualizer.h> //深度图可视化的头文件
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h> //PCL可视化的头文件
#include <pcl/console/parse.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector> // std::vector
#include <string>
using namespace std;
using namespace pcl;
using namespace Eigen;
using namespace cv;
std::vector<std::vector<std::string>> ReadSpaceSeparatedText(std::string filepath)
{
    //打开文件
    std::cout << "Begin reading space separated text file ..." << filepath << std::endl;
    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "fail to open file" << filepath << std::endl;
        exit(1);
    }

    std::vector<std::vector<std::string>> file_content; //存储每个数据的数据结构
    std::string file_line;

    while (std::getline(file, file_line))
    {                                               //读取每一行数据
        std::vector<std::string> file_content_line; //存储分割之后每一行的四个数据
        std::stringstream ss;                       //利用内存流进行每行数据分割内存流
        std::string each_elem;
        ss << file_line;

        while (ss >> each_elem)
        {
            file_content_line.push_back(each_elem);
        }
        file_content.emplace_back(file_content_line); //数据组装
    };
    return file_content;
}
void getpointsparam(Eigen::Vector3d *d_points_W_XYZ, Eigen::Vector3d *d_points_C_XYZ, double *d_points_u, double *d_points_v, bool *d_points_valid, Eigen::Matrix4d W2L, int N, int width, int height, double fx, double fy, double px, double py);
void getimagedata(Eigen::Vector3d *d_faces, unsigned char *d_depth, bool *d_points_valid, double *d_points_u, double *d_points_v, Eigen::Vector3d *d_points_C_XYZ, int M, int width, int height, double fx, double fy, double px, double py, int *face_result);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "usage: ./Demo parameter_setting.yaml" << endl;
        exit(1);
    }

    string yaml_path = argv[1];
    FileStorage Settings(yaml_path, FileStorage::READ);

    if (!Settings.isOpened())
    {
        cerr << "Failed to open settings file at: " << yaml_path << endl;
        exit(1);
    }

    int width = Settings["width"];
    int height = Settings["height"];
    double fx = Settings["fx"];
    double fy = Settings["fy"];
    double px = Settings["px"];
    double py = Settings["py"];
    string ws_folder = Settings["ws_folder"];
    string campose_path = ws_folder + Settings["campose_path"];
    std::vector<std::vector<std::string>> associations = ReadSpaceSeparatedText(ws_folder + Settings["associations"]);
    std::vector<std::vector<std::string>> frameSelected = ReadSpaceSeparatedText(Settings["frame_select_path"]);
    string cloud_path = ws_folder + Settings["cloud_path"];
    string frame_select_path = Settings["frame_select_path"];
    string depth_out_path = ws_folder + Settings["depth_out_path"];
    string color_out_path = ws_folder + Settings["pseudocolor_out_path"];
    const char *depth_out_path_c = depth_out_path.c_str();
    const char *color_out_path_c = color_out_path.c_str();
    string frame_name = Settings["frame_name"];

    Mat rotation;
    Settings["rotation"] >> rotation;

    Mat translation;
    Settings["translation"] >> translation;

    // 读取PLY文件(vertex)获得点云
    PointCloud<PointXYZ>::Ptr point_cloud_ptr(new PointCloud<PointXYZ>);
    if (io::loadPLYFile(cloud_path, *point_cloud_ptr) != 0)
    {
        cout << "Cannot open the file \"" << cloud_path << "\".\n";
        exit(1);
    }
    else
    {
        cout << "points loading success!\n";
    }

    // 读取PLY文件(face)
    PolygonMesh::Ptr mesh_ptr(new PolygonMesh);
    if (io::loadPLYFile(cloud_path, *mesh_ptr) != 0)
    {
        cout << "Cannot open the file \"" << cloud_path << "\".\n";
        exit(1);
    }
    else
    {
        cout << "meshes loading success!\n";
    }

    // 把点云储存成为Vector3d
    int N = point_cloud_ptr->points.size();
    Vector3d *points_W_XYZ = new Vector3d[N];
    for (size_t i = 0; i < point_cloud_ptr->points.size(); i++)
    {
        points_W_XYZ[i](0) = double(point_cloud_ptr->points[i].x);
        points_W_XYZ[i](1) = double(point_cloud_ptr->points[i].y);
        points_W_XYZ[i](2) = double(point_cloud_ptr->points[i].z);
    }

    // 把面片储存成为Vector3d
    int M = mesh_ptr->polygons.size();
    Vector3d *faces = new Vector3d[M];
    for (size_t i = 0; i < mesh_ptr->polygons.size(); i++)
    {
        faces[i](0) = (mesh_ptr->polygons[i]).vertices[0];
        faces[i](1) = (mesh_ptr->polygons[i]).vertices[1];
        faces[i](2) = (mesh_ptr->polygons[i]).vertices[2];
    }

    // RGB和Lucid之间的坐标系转换矩阵
    Matrix4d RGB2Lucid;
    RGB2Lucid(0, 0) = rotation.at<double>(0, 0);
    RGB2Lucid(0, 1) = rotation.at<double>(0, 1);
    RGB2Lucid(0, 2) = rotation.at<double>(0, 2);
    RGB2Lucid(0, 3) = translation.at<double>(0, 0);
    RGB2Lucid(1, 0) = rotation.at<double>(1, 0);
    RGB2Lucid(1, 1) = rotation.at<double>(1, 1);
    RGB2Lucid(1, 2) = rotation.at<double>(1, 2);
    RGB2Lucid(1, 3) = translation.at<double>(1, 0);
    ;
    RGB2Lucid(2, 0) = rotation.at<double>(2, 0);
    RGB2Lucid(2, 1) = rotation.at<double>(2, 1);
    RGB2Lucid(2, 2) = rotation.at<double>(2, 2);
    RGB2Lucid(2, 3) = translation.at<double>(2, 0);
    ;
    RGB2Lucid(3, 0) = 0;
    RGB2Lucid(3, 1) = 0;
    RGB2Lucid(3, 2) = 0;
    RGB2Lucid(3, 3) = 1;
    RGB2Lucid = RGB2Lucid.inverse().eval();

    // 获取每一帧的深度图及伪彩色图像
    Mat depth(height, width, CV_32SC1, Scalar(0));
    Mat depthcolor(height, width, CV_8UC3, Scalar(0));
    Vector3d T;
    Vector4d tmp;
    Vector2d uv;
    int num = 0;
    char name[128];
    char name_color[128];
    int nbytes_XYZ = N * sizeof(Vector3d);
    int nbytes_uv = N * sizeof(double);
    int nbytes_valid = N * sizeof(bool);
    int nbytes_faces = M * sizeof(Vector3d);
    int nbytes_depth = depth.step * depth.rows;
    int nbytes_face_result = depth.cols * depth.rows * sizeof(int);

    //申请device内存
    Vector3d *d_points_W_XYZ;
    Vector3d *d_points_C_XYZ;
    double *d_points_u;
    double *d_points_v;
    bool *d_points_valid;
    int *d_face_result;
    int *face_result;
    face_result = new int[nbytes_face_result];
    cudaMalloc((void **)&d_points_W_XYZ, nbytes_XYZ);
    cudaMalloc((void **)&d_points_C_XYZ, nbytes_XYZ);
    cudaMalloc((void **)&d_points_u, nbytes_uv);
    cudaMalloc((void **)&d_points_v, nbytes_uv);
    cudaMalloc((void **)&d_points_valid, nbytes_valid);
    cudaMalloc((void **)&d_face_result, nbytes_face_result);

    cudaMemcpy((void *)d_points_W_XYZ, (void *)points_W_XYZ, nbytes_XYZ, cudaMemcpyHostToDevice);

    // 申请device内存
    Vector3d *d_faces;
    unsigned char *d_depth;
    cudaMalloc((void **)&d_faces, nbytes_faces);
    cudaMalloc((void **)&d_depth, nbytes_depth);

    cudaMemcpy((void *)d_faces, (void *)faces, nbytes_faces, cudaMemcpyHostToDevice);
    ifstream in_frame_select(frame_select_path, ios::in);
    std::vector<int> frames_selected;
    std::vector<string> frames_selected_str;
    int nowFrameSelected = 0;
    string nowFrameSelected_str;
    int rawNum, depthNum;
    if (in_frame_select.is_open())
    {
        cout << "select frames: " << frame_select_path << " " << endl;
        // while (!in_frame_select.eof())
        // {
        //     in_frame_select >> nowFrameSelected_str;

        //     rawNum = depthNum / 3;
        //     // in_frame_select >> rawNum >> depthNum;
        //     frames_selected.push_back(depthNum * 10000 + rawNum);
        // }
        // cout << "size: " << frames_selected.size() << endl;
    }
    else
    {
        cout << "not select frames" << endl;
    }
    ifstream in(campose_path, ios::in);
    if (!in.is_open())
    {
        cout << "Cannot open the file \"" << campose_path << "\".\n";
        exit(1);
    }
    // sort(frames_selected.begin(), frames_selected.end());
    int now_frameSelected = 0;
    while (!in.eof())
    {
        in >> num;
        in >> T(0) >> T(1) >> T(2) >> tmp(0) >> tmp(1) >> tmp(2) >> tmp(3);
        // if (in_frame_select.is_open())
        {
            string now_frameSelected_str = frameSelected[now_frameSelected][0].substr(frameSelected[now_frameSelected][0].find('1'), 17);

            cout << " " << associations[num][0] << " " << now_frameSelected_str << endl;
            if (associations[num][0] < now_frameSelected_str)
            {
                continue;
            }
            else if (associations[num][0] == now_frameSelected_str)
            {
                now_frameSelected++;
            }
            else
            {
                cout << "some thing wrong: " << associations[num][0] << " " << now_frameSelected_str << endl;
                exit(-1);
            }
        }
        cout << "frame:" << num << endl;

        // 初始化
        for (int row = 0; row < depth.rows; row++)
            for (int col = 0; col < depth.cols; col++)
            {
                depth.at<int>(row, col) = 2147483647;
            }
        cudaMemcpy((void *)d_depth, (void *)(depth.ptr()), nbytes_depth, cudaMemcpyHostToDevice);

        Quaterniond Q(tmp(3), tmp(0), tmp(1), tmp(2));
        Q.normalize();
        Isometry3d W2C(Q);
        W2C.pretranslate(T);
        W2C = W2C.inverse();

        Matrix4d W2L;
        W2L = RGB2Lucid * W2C.matrix();

        getpointsparam(d_points_W_XYZ, d_points_C_XYZ, d_points_u, d_points_v, d_points_valid, W2L, N, width, height, fx, fy, px, py);

        getimagedata(d_faces, d_depth, d_points_valid, d_points_u, d_points_v, d_points_C_XYZ, M, width, height, fx, fy, px, py, face_result);

        cudaMemcpy((void *)(depth.ptr()), (void *)d_depth, nbytes_depth, cudaMemcpyDeviceToHost);

        for (int row = 0; row < depth.rows; row++)
            for (int col = 0; col < depth.cols; col++)
            {
                if (depth.at<int>(row, col) == 2147483647)
                    depth.at<int>(row, col) = 0;
            }

        depth.convertTo(depth, CV_16UC1);
        cout << associations[num][0] << endl;
        sprintf(name, "%s/%s%s.png", depth_out_path_c, frame_name.c_str(), associations[num][0].c_str());
        cout << name << endl;
        imwrite(name, depth);
        depth.convertTo(depth, CV_32SC1);
    }

    cudaFree(d_points_W_XYZ);
    cudaFree(d_points_C_XYZ);
    cudaFree(d_points_u);
    cudaFree(d_points_v);
    cudaFree(d_points_valid);
    cudaFree(d_faces);
    cudaFree(d_depth);

    delete[] faces;
    delete[] points_W_XYZ;

    return 0;
}