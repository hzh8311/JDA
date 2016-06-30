#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/data.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

/*!
 * 简介 Train JoinCascador
 */
void train() {
  Config& c = Config::GetInstance();

  // can we load training data from a binary file
  bool flag = false;

  JoinCascador joincascador;
  joincascador.current_stage_idx = 0;
  joincascador.current_cart_idx = -1;
  c.joincascador = &joincascador; // set global jointcascador

  DataSet pos, neg;
  char data_file[] = "../data/jda_train_data.data";
  if (EXISTS(data_file)) {
    LOG("Load Positive And Negative DataSet from %s", data_file);
    DataSet::Resume(data_file, pos, neg);
  }
  else {
    LOG("Load Positive And Negative DataSet");
    DataSet::LoadDataSet(pos, neg);
    DataSet::Snapshot(pos, neg);
  }

  joincascador.mean_shape = pos.mean_shape;
  LOG("Start training JoinCascador");
  joincascador.Train(pos, neg);
  LOG("End of JoinCascador Training");

  LOG("Saving Model");
  FILE* fd = fopen("../model/jda.model", "wb");
  JDA_Assert(fd, "Can not open the file to save the model");
  joincascador.SerializeTo(fd);
  fclose(fd);
}

/*!
 * \简介 Resume Training Status of JoinCascador
 * \note may not work now
 */
void resume() {
  Config& c = Config::GetInstance();

  FILE* fd = fopen(c.resume_model.c_str(), "rb");
  JDA_Assert(fd, "Can not open model file");

  JoinCascador joincascador;
  c.joincascador = &joincascador; // set global joincascador
  LOG("Loading Model Parameters from model file");
  joincascador.Resume(fd);
  fclose(fd);

  DataSet pos, neg;
  LOG("Load Positive And Negative DataSet");
  DataSet::LoadDataSet(pos, neg);
  joincascador.mean_shape = pos.CalcMeanShape();

  LOG("Forward Positive DataSet");
  DataSet pos_remain;
  const int pos_size = pos.size;
  pos_remain.imgs.reserve(pos_size);
  pos_remain.imgs_half.reserve(pos_size);
  pos_remain.imgs_quarter.reserve(pos_size);
  pos_remain.gt_shapes.reserve(pos_size);
  pos_remain.current_shapes.reserve(pos_size);
  pos_remain.scores.reserve(pos_size);
  pos_remain.last_scores.reserve(pos_size);
  pos_remain.weights.reserve(pos_size);
  // remove tf data points, update score and shape
  for (int i = 0; i < pos_size; i++) {
    int not_used;
    bool is_face = joincascador.Validate(pos.imgs[i], pos.imgs_half[i], pos.imgs_quarter[i], \
                                         pos.scores[i], pos.current_shapes[i], not_used);
    if (is_face) {
      pos_remain.imgs.push_back(pos.imgs[i]);
      pos_remain.imgs_half.push_back(pos.imgs_half[i]);
      pos_remain.imgs_quarter.push_back(pos.imgs_quarter[i]);
      pos_remain.gt_shapes.push_back(pos.gt_shapes[i]);
      pos_remain.current_shapes.push_back(pos.current_shapes[i]);
      pos_remain.scores.push_back(pos.scores[i]);
      pos_remain.last_scores.push_back(0);
      pos_remain.weights.push_back(pos.weights[i]);
    }
  }
  pos_remain.is_pos = true;
  pos_remain.is_sorted = false;
  pos_remain.size = pos_remain.imgs.size();
  neg.Clear();

  LOG("Start Resume Training Status from %dth stage", joincascador.current_stage_idx);
  joincascador.Train(pos_remain, neg);
  LOG("End of JoinCascador Training");

  LOG("Saving Model");
  fd = fopen("../model/jda.model", "wb");
  JDA_Assert(fd, "Can not open the file to save the model");
  joincascador.SerializeTo(fd);
  fclose(fd);
}

void dump() {
  DataSet pos, neg;
  char data_file[] = "../data/jda_train_data.data";
  if (EXISTS(data_file)) {
    LOG("Load Positive And Negative DataSet from %s", data_file);
    DataSet::Resume(data_file, pos, neg);
  }
  pos.Dump("../data/dump/pos");
  neg.Dump("../data/dump/neg");
}
