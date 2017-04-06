// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
// #define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <progress.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;
using namespace arma;
using namespace std;

double PredTransform(double x, const std::string family) {
  if (family == "binomial") {
    return 1 / (1 + exp(-x));
  } else if (family == "gaussian"){
    return x;
  } else if (family == "poisson") {
    return exp(x);
  } else {
    return 0;
  }
}

arma::vec Weight_Compute(double alpha, double beta, double l1, double l2, const arma::vec& z, const arma::vec& n) {
  arma::vec eta = (beta + sqrt(n)) / alpha + l2;
  arma::vec w = (-1 / eta) % (z - sign(z) * l1);
  w.elem(find(abs(z) <= l1)).zeros();
  return w;
}

arma::uvec Dropout_Update(const arma::uvec& ind, double dropout) {
  arma::vec ind_prob = arma::randu<arma::vec>(ind.size());
  return ind.elem(find(ind_prob > dropout));
}

// [[Rcpp::export]]
void set_omp_threads(int n) {
  omp_set_num_threads(n);
}

// [[Rcpp::export]]
int get_omp_threads() {
  return omp_get_max_threads();
}

//' @title FTRL-Proximal Linear Model Predicting Function
//'
//' @description
//' predict_spMatrix predicts values based on linear model weights.
//' This function is an C++ implementation.
//' This function is used internally and is not intended for end-user direct usage.
//'
//' @param x a transposed \code{dgCMatrix} object.
//' @param w a vector of linear model weights.
//' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
//' @return a vector of linear model predicted values.
//' @keywords internal
//' @export
// [[Rcpp::export]]
arma::vec predict_spMatrix(const S4& x, const arma::vec& w, const std::string family) {
  // Design Matrix
  arma::vec x_Dim = x.slot("Dim");
  arma::vec x_p = x.slot("p");
  arma::uvec x_i = x.slot("i");
  arma::vec x_x = x.slot("x");
  //Model Prediction
  arma::vec p(x_Dim[1], fill::zeros);
  // Non-Zero Feature Count for in spMatrix
  arma::vec non_zero_count = diff(x_p);
  // Predicting
  for (int t = 0; t < x_Dim[1]; t++) {
    // Non-Zero Feature Index in spMatrix
    arma::uvec non_zero_index = regspace<arma::uvec>(x_p[t], 1, x_p[t] + non_zero_count[t] - 1);
    // Non-Zero Feature Index for each sample
    arma::uvec i = x_i.elem(non_zero_index);
    // Non-Zero Feature Value for each sample
    arma::vec x_t_i = x_x.elem(non_zero_index);
    // Computing Weight and Prediction
    arma::vec w_i = w.elem(i);
    double p_t = PredTransform(sum(x_t_i % w_i), family);
    // Updating Prediction
    p[t] = p_t;
  }
  // Return Prediction
  return p;
}

//' @title FTRL-Proximal Linear Model Fitting Function
//'
//' @description
//' FTRLProx_train_spMatrix estimates the weights of linear model using FTRL-Proximal Algorithm.
//' This function is an C++ implementation.
//' This function is used internally and is not intended for end-user direct usage.
//'
//' @param x a transposed \code{dgCMatrix}.
//' @param y a vector containing labels.
//' @param state a previously built model state to continue the training from.
//' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
//' @param params a list of parameters of FTRL-Proximal Algorithm.
//' \itemize{
//'   \item \code{alpha} alpha in the per-coordinate learning rate
//'   \item \code{beta} beta in the per-coordinate learning rate
//'   \item \code{l1} L1 regularization parameter
//'   \item \code{l2} L2 regularization parameter
//'   \item \code{dropout} percentage of the input features to drop from each sample
//' }
//' @param epoch The number of iterations over training data to train the model.
//' @param nthread number of parallel threads used to run ftrl.
//' @param verbose logical value. Indicating if the progress bar is displayed or not.
//' @return a list of ftrl model state.
//' @keywords internal
//' @export
// [[Rcpp::export]]
List FTRLProx_train_spMatrix(const S4& x, const arma::vec& y,
                             const Nullable<List>& state, const std::string family, const List params,
                             int epoch, int nthread, bool verbose) {
  // Design Matrix
  arma::vec x_Dim = x.slot("Dim");
  arma::vec x_p = x.slot("p");
  arma::uvec x_i = x.slot("i");
  arma::vec x_x = x.slot("x");
  // Hyperparameter
  double alpha = as<double>(params["alpha"]);
  double beta = as<double>(params["beta"]);
  double l1 = as<double>(params["l1"]);
  double l2 = as<double>(params["l2"]);
  double dropout = as<double>(params["dropout"]);
  //Model Initialization
  arma::vec z(x_Dim[0], fill::zeros);
  arma::vec n(x_Dim[0], fill::zeros);
  arma::vec w(x_Dim[0], fill::zeros);
  if (state.isNotNull()) {
    List state_object = state.get();
    arma::vec state_z = state_object["z"];
    arma::vec state_n = state_object["n"];
    arma::vec state_w = state_object["w"];
    z = state_z;
    n = state_n;
    w = state_w;
  }
  //Model Prediction
  arma::vec p(x_Dim[1], fill::zeros);
  // Non-Zero Feature Count for in spMatrix
  arma::vec non_zero_count = diff(x_p);
  // Initialize Progress Bar
  Progress pb (epoch * x_Dim[1], verbose);
  // Number of Thread
  if (nthread <= 0) {
    nthread = get_omp_threads();
  }
  // Model Updating
  for (int r = 0; r < epoch; r++) {
    if (Progress::check_abort()) {
      return List::create(Rcpp::Named("z") = z,
                          Rcpp::Named("n") = n,
                          Rcpp::Named("w") = w);
    }
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(nthread)
    #endif
    for (unsigned int t = 0; t < y.size(); t++) {
      // Non-Zero Feature Index in spMatrix
      arma::uvec non_zero_index_raw = regspace<arma::uvec>(x_p[t], 1, x_p[t] + non_zero_count[t] - 1);
      // Feature Dropout
      arma::uvec non_zero_index = Dropout_Update(non_zero_index_raw, dropout);
      // Non-Zero Feature Index for each sample
      arma::uvec i = x_i.elem(non_zero_index);
      // Non-Zero Feature Value for each sample
      arma::vec x_t_i = x_x.elem(non_zero_index) / (1 - dropout);
      // Model Parameter
      arma::vec z_i = z.elem(i);
      arma::vec n_i = n.elem(i);
      // Computing Weight and Prediction
      arma::vec w_i = Weight_Compute(alpha, beta, l1, l2, z_i, n_i);
      double p_t = PredTransform(sum(x_t_i % w_i), family);
      // Updating Weight and Prediction
      w.elem(i) = w_i;
      p[t] = p_t;
      // Computing Model Parameter of Next Round
      arma::vec g_i = (p[t] - y[t]) * x_t_i;
      arma::vec s_i = (sqrt(n_i + pow(g_i, 2)) - sqrt(n_i)) / alpha;
      arma::vec z_i_next = z_i + g_i - s_i % w_i;
      arma::vec n_i_next = n_i + pow(g_i, 2);
      // Updating Model Parameter
      z.elem(i) = z_i_next;
      n.elem(i) = n_i_next;
      // Updating Progress Bar
      pb.increment();
    }
  }
  // Retrun FTRL Proximal Model State
  return List::create(Rcpp::Named("z") = z,
                      Rcpp::Named("n") = n,
                      Rcpp::Named("w") = w);
}
