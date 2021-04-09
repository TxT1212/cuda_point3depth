#include <cmath>
#include <vector>
#include <map>
#include <stack>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

struct GeoPoint
{
    double x;    // x坐标
    double y;    // y坐标
    double z;    // z坐标（默认为0，如果需要三维点则给z赋值）

    __device__ GeoPoint(double a = 0, double b = 0, double c = 0) { x = a; y = b; z = c; } // 构造函数
};

struct Line
{
    GeoPoint s;    // 起点
    GeoPoint e;    // 终点
    bool is_seg; // 是否是线段

    __device__ Line() {};    // 默认构造函数
    __device__ Line(GeoPoint a, GeoPoint b, bool _is_seg = true) { s = a; e = b; is_seg = _is_seg; }    // 构造函数(默认是线段)
};

struct Triangle
{
    GeoPoint v0;
    GeoPoint v1;
    GeoPoint v2;
    bool is_plane;

    __device__ Triangle() {}; // 默认构造函数
    __device__ Triangle(GeoPoint a, GeoPoint b, GeoPoint c, bool _is_plane = false) { v0 = a; v1 = b; v2 = c; is_plane = _is_plane; }// 构造函数（默认是三角形）
};


__device__ GeoPoint add(const GeoPoint& lhs, const GeoPoint& rhs)
{
    GeoPoint res;

    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    res.z = lhs.z + rhs.z;

    return res;
}

__device__ GeoPoint sub(const GeoPoint& lhs, const GeoPoint& rhs)
{
    GeoPoint res;

    res.x = lhs.x - rhs.x;
    res.y = lhs.y - rhs.y;
    res.z = lhs.z - rhs.z;

    return res;
}

__device__ GeoPoint mul(const GeoPoint& p, double ratio)
{
    GeoPoint res;

    res.x = p.x * ratio;
    res.y = p.y * ratio;
    res.z = p.z * ratio;

    return res;
}

__device__ GeoPoint div(const GeoPoint& p, double ratio)
{
    GeoPoint res;
    
    res.x = p.x / ratio;
    res.y = p.y / ratio;
    res.z = p.z / ratio;

    return res;
}

__device__ double dotMultiply(const GeoPoint& vec1, const GeoPoint& vec2)
{
    return(vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z);
}

__device__ GeoPoint multiply(const GeoPoint& vec1, const GeoPoint& vec2)
{
    GeoPoint result;

    result.x = vec1.y * vec2.z - vec2.y * vec1.z;
    result.y = vec1.z * vec2.x - vec2.z * vec1.x;
    result.z = vec1.x * vec2.y - vec2.x * vec1.y;

    return result;
}

__device__ double length(const GeoPoint& vec)
{
    return (sqrt(pow(vec.x, 2) + pow(vec.y, 2) + pow(vec.z, 2)));
}

__device__ GeoPoint normalize(const GeoPoint& vec)
{
    GeoPoint res;

    res = div(vec, length(vec));

    return res;
}

__device__ GeoPoint getUnitNormal(const Triangle& t)
{
    GeoPoint vec1 = sub(t.v1, t.v0);
    GeoPoint vec2 = sub(t.v2, t.v0);

    return normalize(multiply(vec1, vec2));
}

__device__ bool isGeoPointInTriangle(const Triangle& t, const GeoPoint& p)
{
    GeoPoint vec1 = sub(t.v1, t.v0);
    GeoPoint vec2 = sub(t.v2, t.v0);
    GeoPoint vec_p = sub(p, t.v0);

    double dot00 = dotMultiply(vec1, vec1);
    double dot01 = dotMultiply(vec1, vec2);
    double dot02 = dotMultiply(vec1, vec_p);
    double dot11 = dotMultiply(vec2, vec2);
    double dot12 = dotMultiply(vec2, vec_p);

    double inverDeno = double(1) / (dot00 * dot11 - dot01 * dot01);
    double u, v;

    u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    if (u < 0 || u > 1) return false;
    if (v < 0 || v > 1) return false;
    if (u + v <= 1)return true;
    else return false;
}

__device__ GeoPoint ltotInterGeoPoint(const Triangle& t, const Line& l)
{
    GeoPoint line_vec = sub(l.e, l.s);
    GeoPoint GeoPoint_vec = sub(t.v0, l.s);
    GeoPoint unit_plane_normal = getUnitNormal(t);

    double ratio = dotMultiply(GeoPoint_vec, unit_plane_normal) / dotMultiply(unit_plane_normal, line_vec);

    return add(l.s, mul(line_vec, ratio));
}

__global__ void getpointsparamDevice(Vector3d* d_points_W_XYZ, Vector3d* d_points_C_XYZ, double* d_points_u, double* d_points_v, bool* d_points_valid, Matrix4d W2L, int N, int width, int height, double fx, double fy, double px, double py)
{
	const int threadID = threadIdx.x + blockIdx.x * blockDim.x;

	if(threadID < N)
	{
		d_points_valid[threadID] = false;

		d_points_C_XYZ[threadID](0) =  W2L(0, 0)*d_points_W_XYZ[threadID](0) + W2L(0, 1)*d_points_W_XYZ[threadID](1) + W2L(0, 2)*d_points_W_XYZ[threadID](2) + W2L(0, 3);
		d_points_C_XYZ[threadID](1) =  W2L(1, 0)*d_points_W_XYZ[threadID](0) + W2L(1, 1)*d_points_W_XYZ[threadID](1) + W2L(1, 2)*d_points_W_XYZ[threadID](2) + W2L(1, 3);
		d_points_C_XYZ[threadID](2) =  W2L(2, 0)*d_points_W_XYZ[threadID](0) + W2L(2, 1)*d_points_W_XYZ[threadID](1) + W2L(2, 2)*d_points_W_XYZ[threadID](2) + W2L(2, 3);


		if(d_points_C_XYZ[threadID](2) > 0)
		{
			d_points_u[threadID] = fx * d_points_C_XYZ[threadID](0) / d_points_C_XYZ[threadID](2) + px;
			d_points_v[threadID] = fy * d_points_C_XYZ[threadID](1) / d_points_C_XYZ[threadID](2) + py;
			if(d_points_u[threadID] >= 0 && d_points_u[threadID] < width && d_points_v[threadID] >= 0 && d_points_v[threadID] < height)
			{
				d_points_valid[threadID] = true;
			}
		}
	}
	else
	{
		return;
	}
	
}

void getpointsparam(Eigen::Vector3d* d_points_W_XYZ, Eigen::Vector3d* d_points_C_XYZ, double* d_points_u, double* d_points_v, bool* d_points_valid, Eigen::Matrix4d W2L, int N, int width, int height, double fx, double fy, double px, double py)
{
	dim3 blocksize(256);
	dim3 gridsize((N + blocksize.x - 1) / blocksize.x);

	getpointsparamDevice <<< gridsize, blocksize >>>(d_points_W_XYZ, d_points_C_XYZ, d_points_u, d_points_v, d_points_valid, W2L, N, width, height, fx, fy, px, py);
    cudaDeviceSynchronize();
}

__global__ void getimagedataDevice(Eigen::Vector3d* d_faces, unsigned char* d_depth, bool* d_points_valid, double* d_points_u, double* d_points_v, Eigen::Vector3d* d_points_C_XYZ, int M, int width, int height, double fx, double fy, double px, double py, int* face_result)
{
	const int threadID = threadIdx.x + blockIdx.x * blockDim.x;

	if(threadID < M)
	{
		bool point1, point2, point3;
		int point_num1, point_num2, point_num3;

		point_num1 = d_faces[threadID](0);
		point_num2 = d_faces[threadID](1);
		point_num3 = d_faces[threadID](2);

		point1 = d_points_valid[point_num1];
		point2 = d_points_valid[point_num2];
		point3 = d_points_valid[point_num3];

		if((point1 || point2 || point3) && (d_points_C_XYZ[point_num1](2) > 0) && (d_points_C_XYZ[point_num2](2) > 0) && (d_points_C_XYZ[point_num3](2) > 0))
		{
			double max_u, min_u, max_v, min_v;
			max_u = max(d_points_u[point_num1], max(d_points_u[point_num2], d_points_u[point_num3]));
			max_v = max(d_points_v[point_num1], max(d_points_v[point_num2], d_points_v[point_num3]));
			min_u = min(d_points_u[point_num1], min(d_points_u[point_num2], d_points_u[point_num3]));
			min_v = min(d_points_v[point_num1], min(d_points_v[point_num2], d_points_v[point_num3]));

			for(int u = int(min_u); u <= (int(max_u) + 1); u++){
                for(int v = int(min_v); v <= (int(max_v) + 1); v++)
                {
					if(u >= 0 && v >= 0 && u < width && v < height)
					{
						GeoPoint p1(d_points_u[point_num1], d_points_v[point_num1], 0);
						GeoPoint p2(d_points_u[point_num2], d_points_v[point_num2], 0);
						GeoPoint p3(d_points_u[point_num3], d_points_v[point_num3], 0);
						GeoPoint p(int(u), int(v), 0);
						Triangle t_plane(p1, p2, p3);

						if(isGeoPointInTriangle(t_plane, p))
						{
							GeoPoint linepoint1((double(u) - px)/fx, (double(v) - py)/fy, 1);
							GeoPoint linepoint2(0, 0, 0);
							GeoPoint joint;
							int pointnum_array[3] = {point_num1, point_num2, point_num3};
							GeoPoint point_array[3];


							for(int i = 0; i < 3; i++)
                            {
                                point_array[i].x = d_points_C_XYZ[pointnum_array[i]](0);
                                point_array[i].y = d_points_C_XYZ[pointnum_array[i]](1);
                                point_array[i].z = d_points_C_XYZ[pointnum_array[i]](2);  
                            }

                            Triangle t_3d(point_array[0], point_array[1], point_array[2]);

							Line l1(linepoint2, linepoint1, false);
							joint = ltotInterGeoPoint(t_3d, l1);

							if(joint.z > 0)
							{
                                int old = atomicMin((int*)(d_depth + 4 * (width*v + u)), (int)(joint.z * 1000));
                                if(old > joint.z * 1000)
                                {
                                    face_result[width*v + u] = threadID;
                                }

							}
						}
					}
                }
            }
        }

	}
	else
	{
		return;
	}

}

void getimagedata(Eigen::Vector3d* d_faces, unsigned char* d_depth, bool* d_points_valid, double* d_points_u, double* d_points_v, Eigen::Vector3d* d_points_C_XYZ, int M, int width, int height, double fx, double fy, double px, double py, int* face_result)
{
	dim3 blocksize(256);
	dim3 gridsize((M + blocksize.x - 1) / blocksize.x);

	getimagedataDevice <<< gridsize, blocksize >>>(d_faces, d_depth, d_points_valid, d_points_u, d_points_v, d_points_C_XYZ, M, width, height, fx, fy, px, py,  face_result);
    cudaDeviceSynchronize();
}


