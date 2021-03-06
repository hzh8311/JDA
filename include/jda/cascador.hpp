#ifndef CASCADOR_HPP_
#define CASCADOR_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace jda {

// forward declaration
class DataSet;
class BoostCart;

/*! 简介: statisic of Detection */
class DetectionStatisic {
public:
  DetectionStatisic()
    : patch_n(0), face_patch_n(0), nonface_patch_n(0), cart_gothrough_n(0.) {
  }

  int patch_n;
  int face_patch_n;
  int nonface_patch_n;
  double cart_gothrough_n;
  double average_cart_n; // nonface patch reject length
};

/*!
 * 简介: JoinCascador for face classification and landmark regression
 */
class JoinCascador {
  /*提供train,Snapshot,Resume,SerializeTo,SerializeFrom方法*/
public:
  /*!
   * 简介: default constructor
   *  This will establish the whole model with parameters from config
   */
  JoinCascador();
  ~JoinCascador();

public:
  /*!
   * 简介: Train JoinCascador
   *  See Full Algorithm on paper `Algorithm 3`
   */
  void Train(DataSet& pos, DataSet& neg);
  /*!
   * 简介: Snapshot current model
   *  Save all model parameters currently trained, this will be used for `Resume`.
   *  A model file will be save at `../model/jda_tmp_{%time%}_{stage}.model`,
   *  `%time%` is the saving time point format `%Y%m%D-%H%M%S` like `20151011-103625`
   *  `stage` is current stage range in [1..c.T]
   */
  void Snapshot() const;
  /*!
   * 简介: Resume the Training Status
   *  Resume the Training Status from a Snapshot model file. We will load the model
   *  parameters and check if the global parameters are the same as `Config`, if not,
   *  the program will be terminateed. After the parameters loaded, Positive Samples
   *  will be processed to generate the Positive DataSet, the Negative DataSet will
   *  be generated by hard negative mining.
   *
   * \param fd      file discriptor of the model file
   */
  void Resume(FILE* fd);
  /*!
   * 简介: Write parameters to a binary file
   * \param   file discriptor of the model file
   */
  void SerializeTo(FILE* fd) const;
  /*!
   * 简介: Read parameters from a binary file
   * \param fd    file discriptor of the model file
   */
  void SerializeFrom(FILE* fd);

public:
  /*!
   * 简介: Validate a region whether a face or not
   *  In training state, we use this function for hard negative mining based on
   *  the training status. In testing state, we just go through all carts to get
   *  a face score for this region. The training status is based on `current_stage_idx`
   *  and `current_cart_idx`.
   *
   * \param img     img
   * \param img_h   half img
   * \param img_q   quarter img
   * \param score   classification score of this image
   * \param shape   shape on this image
   * \param n       number of carts the image go through
   * \return        whether a face or not
   */
  bool Validate(const cv::Mat& img, const cv::Mat& img_h, const cv::Mat& img_q, \
                double& score, cv::Mat_<double>& shape, int& n) const;
  /*!
   * 简介: Detect faces in a gray image
   *  Currently using Sliding Window to search face regions and Non-Maximum Suppression
   *  to group the regions, represented by cv::Rect. All shapes will be relocated in the
   *  original image.
   *
   * \note the interface may change later
   *
   * \param img       gray image
   * \param rects     face locations
   * \param scores    score of faces
   * \param shapes    shape of faces
   * \param statisic  statisic of detection
   * \return          number of faces
   */
  int Detect(const cv::Mat& img, std::vector<cv::Rect>& rects, std::vector<double>& scores, \
             std::vector<cv::Mat_<double> >& shapes, DetectionStatisic& statisic) const;

public:
  /*! 简介: number of stages */
  int T;
  /*! 简介: number of carts */
  int K;
  /*! 简介: number of landmarks */
  int landmark_n;
  /*! 简介: depth of a cart tree */
  int tree_depth;
  /*! 简介: mean shape of positive training data */
  cv::Mat_<double> mean_shape;
  /*! 简介: carts */
  std::vector<BoostCart> btcarts;

  /*!
   * 简介: training status
   *  we have trained the model to current_stage_idx and current_cart_idx
   *  (current_stage_idx, current_stage_idx) = (2, 99) means we have done with stage 0, 1
   *  we are currently on stage 2, we also have done with cart 0, 1, ..., 99. And we are about to
   *  train 100th cart. If (weak classifier/regressor)K == 100, that means we have done with stage 2, and we will do global
   *  regression immediately. When `Snapshot` happens, these two variable will be saved to
   *  the model file to indicate current training status and will be used for further training
   *  or testing. (2, 99) for example. The model file will be `jda_xxxx_stage_3_cart_100.model`.
   *  If K != 100, which means we have not done with this stage, (2, 99) will be saved.
   *  If K == 100, which means we have done with this stage, (3, -1) will be saved, we assume the
   *  global regression has been trained.
   */
  int current_stage_idx;
  int current_cart_idx;
  /*! 简介: training data */
  DataSet* pos;
  DataSet* neg;
};

} // namespace jda

#endif // CASCADOR_HPP_
