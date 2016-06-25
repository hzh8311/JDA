#ifndef DATA_HPP_
#define DATA_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace jda {

// forward declaration
class Cart;
class DataSet;
class Feature;
class JoinCascador;

/*!
 * 简介: Negative Training Sample Generator
 *  hard negative training sample will be needed if less negative alives
 */
class NegGenerator {
public:
  NegGenerator();
  ~NegGenerator();

public:
  /*!
   * 简介: Generate more negative samples
   *  We will generate negative training samples from origin images, all generated samples
   *  should be hard enough to get through all stages of Join Cascador in current training
   *  state, it may be very hard to generate enough hard negative samples, we may fail with
   *  real size smaller than `int size`. We will give back all negative training samples with
   *  their scores and current shapes for further training.
   *
   * \note OpenMP supported hard negative mining, we may have `real size` > `size`
   *
   * \param join_cascador   JoinCascador in training
   * \param size            how many samples we need
   * \param imgs            negative samples
   * \param scores          scores of negative samples
   * \param shapes          shapes of samples, for training
   * \return                real size
   */
  int Generate(const JoinCascador& joincascador, int size, \
               std::vector<cv::Mat>& imgs, std::vector<double>& scores, \
               std::vector<cv::Mat_<double> >& shapes);
  /*!
   * 简介: Load nagetive image file list from path
   * \param path    background image file list
   */
  void Load(const std::vector<std::string>& path);

public:
  /*! 简介: background image list */
  std::vector<std::string> list;
  int current_idx;
  /*! 简介: hard negative list */
  std::vector<cv::Mat> hds;
  int current_hd_idx;
  int times;
  /*! 简介: augment */
  int should_flip;
  int rotation_angle;
  int reset_times;
};

/*!
 * 简介: DataSet Wrapper
 */
class DataSet {
public:
  DataSet();
  ~DataSet();

public:
  /*!
   * 简介: Load Postive DataSet
   *  All positive samples are listed in this text file with each line represents a sample.
   *  We assume all positive samples are processed and generated before our program runs,
   *  this including resize the training samples, grayscale and data augmentation
   *
   * \param positive    a text file path
   */
  void LoadPositiveDataSet(const std::string& positive);
  /*!
   * 简介: Load Negative DataSet
   *  We generate negative samples like positive samples before the program runs. Each line
   *  of the text file hold another text file which holds the real negative sample path in
   *  the filesystem, in this way, we can easily add more negative sample groups without
   *  touching other groups
   *
   * \param negative    negative text list
   */
  void LoadNegativeDataSet(const std::vector<std::string>& negative);
  /*!
   * 简介: Wrapper for `LoadPositiveDataSet` and `LoadNegative DataSet`
   *  Since positive dataset and negative dataset may share some information between
   *  each other, we need to load them all together
   */
  static void LoadDataSet(DataSet& pos, DataSet& neg);
  /*!
   * 简介: Calculate feature values from `feature_pool` with `idx`
   *
   * \param feature_pool    features
   * \param idx             index of dataset to calculate feature value
   * \return                every row presents a feature with every colum presents a data point
   *                        `feature_{i, j} = f_i(data_j)`
   */
  cv::Mat_<int> CalcFeatureValues(const std::vector<Feature>& feature_pool, \
                                  const std::vector<int>& idx) const;
  /*!
   * 简介: Calcualte shape residual of landmark_id over positive dataset
   *  If a landmark id is given, we only generate the shape residual of that landmark
   * \param idx           index of positive dataset
   * \param landmark_id   landmark id to calculate shape residual
   * \return              every data point in each row
   */
  cv::Mat_<double> CalcShapeResidual(const std::vector<int>& idx) const;
  cv::Mat_<double> CalcShapeResidual(const std::vector<int>& idx, int landmark_id) const;
  /*!
   * \biref Calculate Mean Shape over gt_shapes
   * \return    mean_shape of gt_shapes in positive dataset
   */
  cv::Mat_<double> CalcMeanShape();
  /*!
   * 简介: Random Shapes, a random perturbations on mean_shape
   * \param mean_shape    mean shape of positive samples
   * \param shape         random shape
   * \param shapes        this vector should already malloc memory for shapes
   */
  static void RandomShape(const cv::Mat_<double>& mean_shape, cv::Mat_<double>& shape);
  static void RandomShapes(const cv::Mat_<double>& mean_shape, std::vector<cv::Mat_<double> >& shapes);
  /*!
   * 简介: Update weights
   *  `w_i = e^{-y_i*f_i}`, see more on paper in section 4.2
   */
  void UpdateWeights();
  static void UpdateWeights(DataSet& pos, DataSet& neg);
  /*!
   * 简介: Update scores by cart
   *  `f_i = f_i + Cart(x, s)`, see more on paper in `Algorithm 3`
   */
  void UpdateScores(const Cart& cart);
  /*!
   * 简介: Calculate threshold which seperate scores in two part
   *  `sum(scores < th) / N = rate`
   */
  double CalcThresholdByRate(double rate);
  double CalcThresholdByNumber(int remove);
  /*!
   * 简介: Adjust DataSet by removing scores < th
   * \param th    threshold
   */
  void Remove(double th);
  /*!
   * 简介: Get removed number if we perform remove operation
   * \param th    threshold
   */
  int PreRemove(double th);
  /*!
   * 简介: Swap data point
   */
  void Swap(int i, int j);
  /*!
   * 简介: More Negative Samples if needed (only neg dataset needs)
   * \param pos_size    positive dataset size, reference for generating
   * \param rate        N(negative) / N(positive)
   */
  void MoreNegSamples(int pos_size, double rate);
  /*!
   * 简介: Quick Sort by scores descending
   */
  void QSort();
  void _QSort_(int left, int right);
  /*!
   * 简介: Reset score to last_score
   */
  void ResetScores();
  /*!
   * 简介: Clear all
   */
  void Clear();
  /*!
   * 简介: Snapshot all data into a binary file for Resume() maybe
   * \param   pos
   * \param   neg
   */
  static void Snapshot(const DataSet& pos, const DataSet& neg);
  /*!
   * 简介: Resume data from a binary file generated by Snapshot
   * \note  it is useful to generate a binary file for training data which
   *        the load process may cost too much time if your data is very big
   *
   * \param data_file   data file path
   * \param pos         positive dataset
   * \param neg         negative dataset
   */
  static void Resume(const std::string& data_file, DataSet& pos, DataSet& neg);
  /*!
   * 简介: Dump images to file system
   */
  void Dump(const std::string& dir) const;

public:
  /*! 简介: generator for more negative samples */
  NegGenerator neg_generator;
  /*! 简介: face/none-face images */
  std::vector<cv::Mat> imgs;
  std::vector<cv::Mat> imgs_half;
  std::vector<cv::Mat> imgs_quarter;
  // all shapes follows (x_1, y_1, x_2, y_2, ... , x_n, y_n)
  /*! 简介: ground-truth shapes for face */
  std::vector<cv::Mat_<double> > gt_shapes;
  /*! 简介: current shapes */
  std::vector<cv::Mat_<double> > current_shapes;
  /*! 简介: scores, see more about `f_i` on paper */
  std::vector<double> scores;
  std::vector<double> last_scores;
  /*! 简介: weights, see more about `w_i` on paper */
  std::vector<double> weights;
  /*! 简介: is positive dataset */
  bool is_pos;
  /*! 简介: mean shape of positive dataset */
  cv::Mat_<double> mean_shape;
  /*! 简介: is sorted by scores */
  bool is_sorted;
  /*! 简介: size of dataset */
  int size;
};

} // namespace jda

#endif // DATA_HPP_
