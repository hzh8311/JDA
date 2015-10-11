#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/common.hpp"

using namespace cv;
using namespace std;

namespace jda {

int Feature::CalcFeatureValue(const Mat& o, const Mat& h, const Mat& q, \
                              const Mat_<double>& s) const {
    double ratio;
    int height, width;
    Mat img;
    switch (scale) {
    case ORIGIN:
        ratio = 1;
        height = o.rows;
        width = o.cols;
        img = o; // ref
        break;
    case HALF:
        ratio = double(h.rows) / double(o.rows);
        height = h.rows;
        width = h.cols;
        img = h; // ref
        break;
    case QUARTER:
        ratio = double(q.rows) / double(o.rows);
        height = q.rows;
        width = q.cols;
        img = q; // ref
        break;
    default:
        dieWithMsg("Unsupported SCALE");
        break;
    }

    double x1, y1, x2, y2;
    x1 = s(0, 2 * landmark_id1) + o.cols*offset1_x;
    y1 = s(0, 2 * landmark_id1 + 1) + o.rows*offset1_y;
    x2 = s(0, 2 * landmark_id2) + o.cols*offset2_x;
    y2 = s(0, 2 * landmark_id2 + 1) + o.rows*offset2_y;
    x1 *= ratio; y1 *= ratio;
    x2 *= ratio; y2 *= ratio;
    int x1_ = int(round(x1));
    int y1_ = int(round(y1));
    int x2_ = int(round(x2));
    int y2_ = int(round(y2));

    checkBoundaryOfImage(width, height, x1_, y1_);
    checkBoundaryOfImage(width, height, x2_, y2_);

    int val = img.at<uchar>(x1_, y1_) - img.at<uchar>(x2_, y2_);
    return val;
}

void LOG(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char msg[256];
    vsprintf(msg, fmt, args);
    va_end(args);

    char buff[256];
    time_t t = time(NULL);
    strftime(buff, sizeof(buff), "[%x - %X]", localtime(&t));
    printf("%s %s\n", buff, msg);
}

void dieWithMsg(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char msg[256];
    vsprintf(msg, fmt, args);
    va_end(args);

    LOG(msg);
    exit(-1);
}

double calcVariance(const Mat_<double>& vec) {
    double m1 = cv::mean(vec)[0];
    double m2 = cv::mean(vec.mul(vec))[0];
    double variance = m2 - m1*m1;
    return variance;
}
double calcVariance(const vector<double>& vec) {
    if (vec.size() == 0) return 0.;
    Mat_<double> vec_(vec);
    double m1 = cv::mean(vec_)[0];
    double m2 = cv::mean(vec_.mul(vec_))[0];
    double variance = m2 - m1*m1;
    return variance;
}

double calcMeanError(const vector<Mat_<double> >& gt_shapes, \
                     const vector<Mat_<double> >& current_shapes) {
    const Config& c = Config::GetInstance();
    const int N = gt_shapes.size();
    const int landmark_n = c.landmark_n;
    double e = 0.;
    Mat_<double> delta_shape;
    for (int i = 0; i < N; i++) {
        delta_shape = gt_shapes[i] - current_shapes[i];
        for (int j = 0; j < landmark_n; j++) {
            e += std::sqrt(std::pow(delta_shape(0, 2 * j), 2) + \
                           std::pow(delta_shape(0, 2 * j + 1), 2));
        }
    }
    e /= landmark_n * N;
    e /= c.img_o_width;
    return e;
}

Config::Config() {
    T = 5;
    K = 1080;
    landmark_n = 5;
    tree_depth = 4;
    accept_rate = 0.9999;
    reject_rate = 0.3;
    shift_size = 10;
    np_ratio = 1;
    img_o_height = img_o_width = 80;
    img_h_height = img_h_width = 56;
    img_q_height = img_q_width = 40;
    x_step = y_step = 20;
    scale_factor = 0.8;
    int feats[5] = { 500, 500, 500, 300, 300 };
    double radius[5] = { 0.4, 0.3, 0.2, 0.15, 0.1 };
    double probs[5] = { 0.9, 0.8, 0.7, 0.6, 0.5 };
    this->feats.clear();
    this->radius.clear();
    this->probs.clear();
    for (int i = 0; i < T; i++) {
        this->feats.push_back(feats[i]);
        this->radius.push_back(radius[i]);
        this->probs.push_back(probs[i]);
    }
    train_txt = "../data/train.txt";
    test_txt = "../data/test.txt";
    nega_txt = "../data/nega.txt";
}

} // namespace jda