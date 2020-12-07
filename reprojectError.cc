#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>       // std::vector
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <glob.h>
#include <opencv2/imgproc.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <numeric>   
using namespace std;
using namespace cv;
int main(int argc, char** argv)
{
        if(argc != 3)
    {
        cout << "usage: ./Demo parameter_setting.yaml parameter_settinghd.yaml" << endl;
        exit(1);
    }
    string yaml_path = argv[1];
    FileStorage Settings(yaml_path, FileStorage::READ);
    string yaml_path1 = argv[2];
    FileStorage Settings1(yaml_path1, FileStorage::READ);    

    string ws_folder = Settings1["ws_folder"];
    string frame_select_path = Settings1["frame_select_path"];
    string depth_out_path = ws_folder + Settings["depth_out_path"];
    string depth_out_path1 = ws_folder + Settings1["depth_out_path"];

    
    ifstream in_frame_select(frame_select_path, ios::in);
    std::vector<int> frames_selected;
    std::vector<int> rawframes_selected;
    int nowFrameSelected = 0;
    char name[128];
    int rawNum, depthNum;
    if(in_frame_select.is_open())
    {
        cout << "select frames: " << frame_select_path << " ";
        while(!in_frame_select.eof())
        {
            in_frame_select >> rawNum >> depthNum;
            frames_selected.push_back(depthNum);
            rawframes_selected.push_back(rawNum);
        }
        cout << "size: " << frames_selected.size() << endl;
    }
    else
    {
        cout << "not select frames" << endl;
        exit(-1);
    }
    vector<double> errorAvgV;
    vector<double> errorStdV;
    errorAvgV.reserve(frames_selected.size());
    errorStdV.reserve(frames_selected.size());
    for (int ii = 0; ii < frames_selected.size(); ii++)
    {
        sprintf(name, "/frame_%05d.png", frames_selected[ii]);
        Mat depthMat = imread(depth_out_path + name, CV_LOAD_IMAGE_UNCHANGED);
        cout << depth_out_path + name << " " << depthMat.cols<< endl;
        sprintf(name, "/frame_%05d.png", rawframes_selected[ii]);
        Mat depthMatHD = imread(depth_out_path1 + name, CV_LOAD_IMAGE_UNCHANGED);
        vector<double> depthDelta;
        depthDelta.reserve(depthMat.cols * depthMat.rows);
        cout << depth_out_path1 + name <<" " << depthMatHD.cols<<  endl;
        for (int i = 1; i < depthMat.rows-1; i++)
        {
            for (int j = 1; j< depthMat.cols-1; j++)
            {
                double depthRaw =  depthMat.at<ushort>(i,j);
                if (depthRaw == 0|| depthMatHD.at<ushort>(i*3-1,j*3) == 0|| depthMatHD.at<ushort>(i*3+1,j*3) == 0|| depthMatHD.at<ushort>(i*3,j*3-1) == 0|| depthMatHD.at<ushort>(i*3,j*3+1) == 0)
                {
                    continue;
                }
                double depthHD = depthMatHD.at<ushort>(i*3-1,j*3)+depthMatHD.at<ushort>(i*3+1,j*3)+depthMatHD.at<ushort>(i*3,j*3-1)+depthMatHD.at<ushort>(i*3+1,j*3+1);
                // double depthHD = max(max(depthMatHD.at<ushort>(i*3-1,j*3),depthMatHD.at<ushort>(i*3+1,j*3)),max(depthMatHD.at<ushort>(i*3,j*3-1),depthMatHD.at<ushort>(i*3+1,j*3+1)));
                
                depthDelta.push_back(abs(depthHD/4. - depthRaw));
                // depthDelta.push_back(abs(depthHD - depthRaw));
                // cout << depthHD/4 << " " << depthRaw << endl;
            }
        }
        // cout << "f" << endl;

        double errorAvg = std::accumulate(depthDelta.begin(), depthDelta.end(), 0)*1.0/depthDelta.size();
        errorAvgV.push_back(errorAvg);
        double errorStd = 0;
        for(auto iter: depthDelta)
        {
            errorStd += pow(iter - errorAvg, 2);
        }
        errorStd /= depthDelta.size();
        errorStd = sqrt(errorStd);
        errorStdV.push_back(errorStd);
        cout << errorAvg << " " << errorStd << " " << depthDelta.size() << " " << std::accumulate(depthDelta.begin(), depthDelta.end(), 0) << endl;
    }
    cout << "errorStdV: " << accumulate(errorStdV.begin(), errorStdV.end(), 0)*1.0/errorStdV.size() << " " << errorStdV[errorStdV.size()/2] << endl;
    cout << "errorAvgV: " << accumulate(errorAvgV.begin(), errorAvgV.end(), 0)*1.0/errorAvgV.size() << " " << errorAvgV[errorAvgV.size()/2] << endl;
}