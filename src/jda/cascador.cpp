#include <ctime>
#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/jda.hpp"

using namespace cv;
using namespace std;

namespace jda {

static int YO = 0;

JoinCascador::JoinCascador() {}
JoinCascador::~JoinCascador() {}
void JoinCascador::Initialize(int T) {
    this->T = T;
    btcarts.resize(T);
    for (int t = 0; t < T; t++) {
        btcarts[t].Initialize(t);
        btcarts[t].set_joincascador(this);
    }
}

void JoinCascador::Train(DataSet& pos, DataSet& neg) {
    for (int t = 0; t < T; t++) {
        current_stage_idx = t;
        current_cart_idx = -1;
        LOG("Train %d th stages", t + 1);
        TIMER_BEGIN
            btcarts[t].Train(pos, neg);
            LOG("End of train %d th stages, costs %.4lf s", t + 1, TIMER_NOW);
        TIMER_END
        LOG("Snapshot current Training Status");
        Snapshot();
    }
}

void JoinCascador::Snapshot() {
    int stage = current_stage_idx + 1;
    char buff1[256];
    char buff2[256];
    time_t t = time(NULL);
    strftime(buff1, sizeof(buff1), "%Y%m%d-%H%M%S", localtime(&t));
    sprintf(buff2, "../model/jda_tmp_%s_stage%d.model", buff1, stage);

    FILE* fd = fopen(buff2, "wb");
    JDA_Assert(fd, "Can not open a temp file to save the model");
    SerializeTo(fd, stage);
    fclose(fd);
}
void JoinCascador::Resume(int stage, FILE* fd) {
    // **TODO** Resume
    current_stage_idx = stage - 1;
    current_cart_idx = -1;
}

void JoinCascador::SerializeTo(FILE* fd, int stage) {
    const Config& c = Config::GetInstance();
    if (stage <= 0 || stage > c.T) stage = c.T;

    fwrite(&YO, sizeof(YO), 1, fd);
    fwrite(&stage, sizeof(int), 1, fd);
    fwrite(&c.K, sizeof(int), 1, fd);
    fwrite(&c.landmark_n, sizeof(int), 1, fd);
    fwrite(&c.tree_depth, sizeof(int), 1, fd);
    // btcarts
    for (int t = 0; t < stage; t++) {
        const BoostCart& btcart = btcarts[t];
        for (int k = 0; k < c.K; k++) {
            const Cart& cart = btcart.carts[k];
            // only non leaf node need to save parameters
            for (int i = 0; i < cart.nodes_n / 2; i++) {
                const Feature& feature = cart.features[i];
                fwrite(&feature.scale, sizeof(int), 1, fd);
                fwrite(&feature.landmark_id1, sizeof(int), 1, fd);
                fwrite(&feature.landmark_id2, sizeof(int), 1, fd);
                fwrite(&feature.offset1_x, sizeof(double), 1, fd);
                fwrite(&feature.offset1_y, sizeof(double), 1, fd);
                fwrite(&feature.offset2_x, sizeof(double), 1, fd);
                fwrite(&feature.offset2_y, sizeof(double), 1, fd);
            }
            // leaf node has scores
            for (int i = 0; i < cart.nodes_n / 2; i++) {
                fwrite(&cart.scores[i], sizeof(double), 1, fd);
            }
            // threshold
            fwrite(&cart.th, sizeof(double), 1, fd);
        }
        // global regression parameters
        const double* w_ptr;
        const int rows = btcart.w.rows;
        const int cols = btcart.w.cols;
        for (int i = 0; i < rows; i++) {
            w_ptr = btcart.w.ptr<double>(i);
            fwrite(w_ptr, sizeof(double), cols, fd);
        }
    }
    fwrite(&YO, sizeof(YO), 1, fd);
}

void JoinCascador::SerializeFrom(FILE* fd, int stage) {
    const Config& c = Config::GetInstance();
    if (stage <= 0 || stage > c.T) stage = c.T;

    int tmp;
    fread(&YO, sizeof(YO), 1, fd);
    fread(&tmp, sizeof(int), 1, fd);
    if (tmp != stage) dieWithMsg("Wrong Model Paratemers!");
    fread(&tmp, sizeof(int), 1, fd);
    if (tmp != c.K) dieWithMsg("Wrong Model Paratemers!");
    fread(&tmp, sizeof(int), 1, fd);
    if (tmp != c.landmark_n) dieWithMsg("Wrong Model Paratemers!");
    fread(&tmp, sizeof(int), 1, fd);
    if (tmp != c.tree_depth) dieWithMsg("Wrong Model Paratemers!");

    current_stage_idx = c.T - 1;
    current_cart_idx = c.K - 1;

    btcarts.resize(c.T); // still need to malloc full memory
    for (int t = 0; t < stage; t++) {
        BoostCart& btcart = btcarts[t];
        btcart.Initialize(t);
        for (int k = 0; k < c.K; k++) {
            Cart& cart = btcart.carts[k];
            cart.Initialize(t, k%c.landmark_n);
            // only non leaf node need to save parameters
            for (int i = 0; i < cart.nodes_n / 2; i++) {
                Feature& feature = cart.features[i];
                fread(&feature.scale, sizeof(int), 1, fd);
                fread(&feature.landmark_id1, sizeof(int), 1, fd);
                fread(&feature.landmark_id2, sizeof(int), 1, fd);
                fread(&feature.offset1_x, sizeof(double), 1, fd);
                fread(&feature.offset1_y, sizeof(double), 1, fd);
                fread(&feature.offset2_x, sizeof(double), 1, fd);
                fread(&feature.offset2_y, sizeof(double), 1, fd);
            }
            // leaf node has scores
            for (int i = 0; i < cart.nodes_n / 2; i++) {
                fread(&cart.scores[i], sizeof(double), 1, fd);
            }
            // threshold
            fread(&cart.th, sizeof(double), 1, fd);
        }
        // global regression parameters
        double* w_ptr;
        const int rows = c.landmark_n * 2;
        const int cols = c.K * (1 << (c.tree_depth - 1));
        btcart.w.create(rows, cols);
        for (int i = 0; i < rows; i++) {
            w_ptr = btcart.w.ptr<double>(i);
            fread(w_ptr, sizeof(double), cols, fd);
        }
    }
    fread(&YO, sizeof(YO), 1, fd);
}

bool JoinCascador::Validate(const Mat& img, double& score, Mat_<double>& shape) const {
    const Config& c = Config::GetInstance();
    Mat img_h, img_q;
    cv::resize(img, img_h, Size(c.img_h_width, c.img_h_height));
    cv::resize(img, img_q, Size(c.img_q_width, c.img_q_height));
    DataSet::RandomShape(mean_shape, shape);
    score = 0;
    Mat_<int> lbf(1, c.K);
    int* lbf_ptr = lbf.ptr<int>(0);
    const int base = 1 << (c.tree_depth - 1);
    int offset = 0;
    // stage [0, current_stage_idx)
    for (int t = 0; t < current_stage_idx; t++) {
        const BoostCart& btcart = btcarts[t];
        offset = 0;
        for (int k = 0; k < c.K; k++) {
            const Cart& cart = btcart.carts[k];
            int idx = cart.Forward(img, img_h, img_q, shape);
            score += cart.scores[idx];
            if (score < cart.th) {
                // not a face
                return false;
            }
            lbf_ptr[k] = offset + idx;
            offset += base;
        }
        // global regression
        shape += btcart.GenDeltaShape(lbf);
    }
    // current stage
    for (int k = 0; k <= current_cart_idx; k++) {
        const Cart& cart = btcarts[current_stage_idx].carts[k];
        int idx = cart.Forward(img, img_h, img_q, shape);
        score += cart.scores[idx];
        if (score < cart.th) {
            // not a face
            return false;
        }
    }
    return true;
}

} // namespace jda