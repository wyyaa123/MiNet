// #include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
// #include <Eigen/Core>
// #include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

#include "./include/randomNum.hpp"
#include "./include/timeRecorder.hpp"
using namespace std;

void findCornerPoints(cv::InputOutputArray&, cv::InputOutputArray&, const cv::TermCriteria&);

void match(string type, cv::Mat& desc1, cv::Mat& desc2, std::vector<cv::DMatch>& matches);

void detect_and_compute(std::string type, cv::Mat& img, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

void LK_opticalFlow(int argc, char** argv);

void calVO(std::vector<cv::KeyPoint>& kpt1, std::vector<cv::KeyPoint>& kpt2, std::vector<cv::DMatch>& matches, cv::Mat& R, cv::Mat& t, vector<cv::Point3f>& points);

double distance(cv::Point2f &pt1, cv::Point2f &pt2);

cv::Scalar randomColor();

int main (int argc, char** argv) {
    
    // cv::Mat img1 = cv::imread(argv[1], cv::ImreadModes::IMREAD_COLOR);
    // cv::Mat img2 = cv::imread(argv[2], cv::ImreadModes::IMREAD_COLOR);

    // std::vector<cv::KeyPoint> kpts1; 
    // std::vector<cv::KeyPoint> kpts2; 
    // cv::Mat desc1;
    // cv::Mat desc2;
    // cv::Mat res;

    // std::vector<cv::DMatch> matches;

    // clock_t beg = clock();
    // detect_and_compute("ORB", img1, kpts1, desc1);
    // detect_and_compute("ORB", img2, kpts2, desc2);

    // printf("keypoints1: %ld, keypoints2: %ld\n", kpts1.size(), kpts2.size());

    // match("knn", desc1, desc2, matches);

    // double elsp_time = double(clock() - beg) / CLOCKS_PER_SEC;

    // cout << "total elasped time is " << elsp_time << "seconds" << endl;

    // cv::drawMatches(img1, kpts1, img2, kpts2, matches, res, cv::Scalar::all(-1), cv::Scalar::all(255), vector<char>(matches.size(), 1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // cv::imshow("result", res);
    // cv::waitKey();
    // cv::destroyAllWindows();

    // cv::imwrite("./result1.jpg", res);
    LK_opticalFlow(argc, argv);
    
    return 0;
}

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < 480 - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < 752 - BORDER_SIZE;
}

double distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    // printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void findCornerPoints (cv::InputOutputArray& _img, cv::InputOutputArray& _corner_points, const cv::TermCriteria& criteria) {
    cv::Mat gray_img;
    cv::cvtColor(_img, gray_img, cv::COLOR_BGR2GRAY); //使用灰度图进行角点检测。

    cv::goodFeaturesToTrack(gray_img, _corner_points, 100, 0.1, 10); //查找像素级角点

    cv::cornerSubPix(gray_img, _corner_points, cv::Size(5, 5), cv::Size(-1, -1), criteria);///查找亚像素角点
}

void LK_opticalFlow (int argc, char** argv) {

    static TimeRecorder timer = TimeRecorder();

    /*
    void calcOpticalFlowPyrLK (cv::InputArray prevImg, 
                               cv::InputArray nextImg, 
                               cv::InputArray prevPts, 
                               cv::InputOutputArray nextPts, 
                               cv::OutputArray status, 
                               cv::OutputArray err, 
                               cv::Size winSize = cv::Size(21, 21), //跟踪窗口大小
                               int maxLevel = 3, //金字塔层数
                               cv::TermCriteria criteria = cv::TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 
                                                                             30, 
                                                                             (0.01)), //算法终止判断标准
                               int flags = 0, 
                               double minEigThreshold = (0.0001))
    */
    if(argc != 3) {
        cout << "Call: " << argv[0] <<  "[image1] [image2]" << endl;
        cout << "Demonstrates Pyramid Lucas-Kanada optical flow. " << endl;
        exit(-1);
    }

    cv::Mat img1, _img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::cvtColor(_img1, img1, cv::COLOR_BGR2GRAY);

    cv::Mat img2, _img2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    cv::cvtColor(_img2, img2, cv::COLOR_BGR2GRAY);

    cv::Size img_sz = img1.size();
    int win_size = 10;
    cv::Mat img3 = cv::imread(argv[2], cv::IMREAD_COLOR);

    vector<cv::Point2f> prev_pts, cur_pts, reverse_pts;
    const int MAX_CORNERS = 500;
    cv::goodFeaturesToTrack (img1, prev_pts, MAX_CORNERS, .01, 5); //计算改善型的Harris角点

    printf("prev_pts size is %ld\n", prev_pts.size());

    // cv::cornerSubPix(img1, 
    //                  prev_pts, 
    //                  cv::Size(win_size, win_size), 
    //                  cv::Size(-1, -1), 
    //                  cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 20, 0.03)); //亚像素级别角点

    vector<uchar> status, reverse_status;

    cv::calcOpticalFlowPyrLK(img1, img2, 
                             prev_pts, cur_pts, 
                             status, cv::noArray(), //status的每个元素都会提升是否找到了prevPts中的相应特征
                             cv::Size(win_size* 2 + 1, win_size * 2 + 1), 5, 
                             cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 20, 0.03));


    if (1) { // FLOW_BACK
        cv::calcOpticalFlowPyrLK(img2, img1, 
                                 cur_pts, reverse_pts, 
                                 reverse_status, cv::noArray(), //status的每个元素都会提升是否找到了prevPts中的相应特征
                                 cv::Size(win_size* 2 + 1, win_size * 2 + 1), 5, 
                                 cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 20, 0.03));
                                
        for (size_t i = 0; i < status.size(); ++i)
            {
                if (status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.4)
                    status[i] = 1;
                else
                    status[i] = 0;
            }
    }

    int n = 0;
    for (auto& i: status) {
        if (i)
            ++n;
    }
    printf("cur_pts size is %d\n", n);

    for(int i = 0; i != prev_pts.size(); ++i) {
        if(!status[i]) continue;
        cv::line(img3, prev_pts[i], cur_pts[i], randomColor(), 2, cv::LINE_AA);
    }

    cout << "total expend " << timer.get_time() << "seconds" << endl;

    // cv::imshow("img1", _img1);
    // cv::imshow("img2", _img2);
    cv::imshow("LK Optical Flow Example", img3);
    cv::waitKey(0);

    cv::imwrite("./result.jpg", img3);
}

void calVO(std::vector<cv::KeyPoint>& kpt1, std::vector<cv::KeyPoint>& kpt2, 
           std::vector<cv::DMatch>& matches, 
           cv::Mat& R, cv::Mat& t, 
           vector<cv::Point3f>& points,
           vector<char>& match_mask, 
           cv::Point2f principal_point,
           double focal_length) {

    std::vector<cv::Point2f> queryPoints;
    std::vector<cv::Point2f> trainPoints;

    std::for_each(matches.begin(), matches.end(), [&](const cv::DMatch& match) -> void {
        queryPoints.emplace_back(kpt1[match.queryIdx].pt);
        trainPoints.emplace_back(kpt1[match.trainIdx].pt);
    });

    cv::Mat essential_matrix = cv::findEssentialMat(queryPoints, trainPoints, focal_length, principal_point, cv::RANSAC, 1);
    cv::recoverPose( essential_matrix, queryPoints, trainPoints, R, t, focal_length, principal_point );
}

/**
 * @brief 在图像中画点，注意该函数会直接在输入中画点
 * 
 * @param img 输入图像
 * @param points 点集合
 */
cv::Scalar randomColor() {
        auto random1 = getRandomNum(0, 255);
        auto random2 = getRandomNum(0, 255);
        auto random3 = getRandomNum(0, 255);
        return cv::Scalar(random1, random2, random3);
}

/**
 * @brief 计算关键子和描述符
 * 
 * @param type 使用的算法：ORB特征点, FAST角点, BLOB角点, SURF特征点, BRISK特征点, KAZE特征点, AKAZE特征点, FREAK角点, DAISY角点, BRIEF角点。使用时请输入他们的英文缩写
 * @param img 用于提取角点/特征点的图像
 * @param kpts 关键子
 * @param desc 描述符
 */
inline void detect_and_compute(std::string type, cv::Mat& img, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    std::for_each(type.begin(), type.end(), [](char& c) ->void { c = std::tolower(c); });
    if (type.find("fast") == 0) {
        // type = type.substr(4);
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(10, true);
        detector->detect(img, kpts);
    }
    if (type.find("blob") == 0) {
        // type = type.substr(4);
        cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create();
        detector->detect(img, kpts);
    }
    if (type == "surf") {
        cv::Ptr<cv::Feature2D> surf = cv::xfeatures2d::SURF::create(800.0);
        surf->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "sift") {
        cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
        sift->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "orb") {
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "brisk") {
        cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
        brisk->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "kaze") {
        cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
        kaze->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "akaze") {
        cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
        akaze->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "freak") { 
        cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
        freak->compute(img, kpts, desc);
    }
    if (type == "daisy") {
        cv::Ptr<cv::xfeatures2d::DAISY> daisy = cv::xfeatures2d::DAISY::create();
        daisy->compute(img, kpts, desc);
    }
    if (type == "brief") {
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create(64);
        brief->compute(img, kpts, desc);
    }
}

/**
 * @brief 用于特征点匹配
 * 
 * @param type 匹配方法：暴力匹配 bf; k-nearest neighbors(fnn)匹配; Fast Library for Approximate Nearest Neighbors(flann)匹配
 * @param desc1 查询图像中的特征点的描述符
 * @param desc2 目标图像中的特征点的描述符
 * @param matches 输出的匹配结果
 */
inline void match(string type, cv::Mat& desc1, cv::Mat& desc2, std::vector<cv::DMatch>& matches) {

    const double kDistanceCoef = 4.0;
    const int kMaxMatchingSize = 50;

    matches.clear();
    if (type == "bf") {
        cv::BFMatcher desc_matcher(cv::NORM_L2, true); //欧几里得距离
        desc_matcher.match(desc1, desc2, matches, cv::Mat());
    } else if (type == "knn") {
        cv::BFMatcher desc_matcher(cv::NORM_L2, true);
        vector<vector<cv::DMatch>> vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) { if (!vmatches[i].size()) continue;  matches.push_back(vmatches[i][0]); }
    } else if (type == "flann") {
        cv::FlannBasedMatcher flnn_matcher;
        flnn_matcher.match(desc1, desc2, matches);
    }

    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}