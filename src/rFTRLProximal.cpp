// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <RcppArmadillo.h>
#include <progress.hpp>
using namespace Rcpp;
using namespace arma;
using namespace std;

double PredTransform(double x, const std::string family) {
  if (family == "gaussian") {
    return x;
  } else if (family == "binomial"){
    return 1 / (1 + exp(-x));
  } else if (family == "poisson") {
    return exp(x);
  } else {
    return 0;
  }
}

arma::vec Weight_Update(double alpha, double beta, double l1, double l2, arma::vec z, arma::vec n) {
  arma::vec eta = (beta + sqrt(n)) / alpha + l2;
  arma::vec w = (-1 / eta) % (z - sign(z) * l1);
  w.elem(find(abs(z) <= l1)).zeros();
  return w;
}

//' @title FTRL-Proximal Linear Model Predicting Function
//'
//' @description
//' FTRLProx_predict_spMatrix predicts values based on linear model weights.
//' This function is an C++ implementation.
//' This function is used internally and is not intended for end-user direct usage.
//'
//' @param x a transposed \code{dgCMatrix} object.
//' @param w a vector of linear model weights.
//' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
//' @return a vector of linear model predicted values
//' @export
// [[Rcpp::export]]
arma::vec FTRLProx_predict_spMatrix(arma::sp_mat x, arma::vec w, const std::string family) {
  arma::vec p = arma::vectorise(arma::conv_to<arma::rowvec>::from(w) * x);
  if (family == "gaussian") {
    return p;
  } else if (family == "binomial"){
    return 1 / (1 + exp(-p));
  } else if (family == "poisson") {
    return exp(p);
  } else {
    return arma::vec(p.size());
  }
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
//' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
//' @param params a list of parameters of FTRL-Proximal Algorithm.
//' \itemize{
//'   \item \code{alpha} alpha in the per-coordinate learning rate
//'   \item \code{beta} beta in the per-coordinate learning rate
//'   \item \code{l1} L1 regularization parameter
//'   \item \code{l2} L2 regularization parameter
//' }
//' @param epoch The number of iterations over training data to train the model.
//' @param verbose logical value. Indicating if the progress bar is displayed or not.
//' @return a vector of linear model weights
//' @export
// [[Rcpp::export]]
arma::vec FTRLProx_train_spMatrix(S4 x, arma::vec y, const std::string family, List params, int epoch, bool verbose) {
  // Hyperparameter
  double alpha = as<double>(params["alpha"]);
  double beta = as<double>(params["beta"]);
  double l1 = as<double>(params["l1"]);
  double l2 = as<double>(params["l2"]);
  // Design Matrix
  arma::vec x_Dim = x.slot("Dim");
  arma::vec x_p = x.slot("p");
  arma::uvec x_i = x.slot("i");
  arma::vec x_x = x.slot("x");
  //Model Initialization
  arma::vec z(x_Dim[0], fill::zeros);
  arma::vec n(x_Dim[0], fill::zeros);
  arma::vec w(x_Dim[0], fill::zeros);
  // Model Prediction
  arma::vec p(x_Dim[1], fill::zeros);
  // Training and Validation Performance
  arma::vec eval_train(epoch, fill::zeros);
  arma::vec eval_val(epoch, fill::zeros);
  // Non-Zero Feature Count for in spMatrix
  arma::vec non_zero_count = diff(x_p);
  // Initialize Progress Bar
  Progress pb (epoch * x_Dim[1], verbose);
  // Model Updating
  for (int r = 0; r < epoch; r++) {
    if (Progress::check_abort()) {
      return w;
    }
    for (int t = 0; t < y.size(); t++) {
      // Non-Zero Feature Index in spMatrix
      arma::uvec non_zero_index = regspace<arma::uvec>(x_p[t], 1, x_p[t] + non_zero_count[t] - 1);
      // Non-Zero Feature Index for each sample
      arma::uvec i = x_i.elem(non_zero_index);
      // Non-Zero Feature Value for each sample
      arma::vec x_t_i = x_x.elem(non_zero_index);
      // Model Parameter
      arma::vec z_i = z.elem(i);
      arma::vec n_i = n.elem(i);
      // Computing Weight and Prediction
      arma::vec w_i = Weight_Update(alpha, beta, l1, l2, z_i, n_i);
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
  // Retrun FTRL Proximal Model Weight
  return w;
}

//' @title FTRL-Proximal Linear Model Validation Function
//'
//' @description
//' FTRLProx_validate_spMatrix validates the performance of FTRL-Proximal online learning model.
//' This function is an C++ implementation.
//' This function is used internally and is not intended for end-user direct usage.
//'
//' @param x a transposed \code{dgCMatrix}.
//' @param y a vector containing labels.
//' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
//' @param params a list of parameters of FTRL-Proximal Algorithm.
//' \itemize{
//'   \item \code{alpha} alpha in the per-coordinate learning rate
//'   \item \code{beta} beta in the per-coordinate learning rate
//'   \item \code{l1} L1 regularization parameter
//'   \item \code{l2} L2 regularization parameter
//' }
//' @param epoch The number of iterations over training data to train the model.
//' @param val_x a transposed \code{dgCMatrix} for validation.
//' @param val_y a vector containing labels for validation.
//' @param eval a evaluation metrics computing function, the first argument shoule be prediction, the second argument shoule be label.
//' @param patience The number of rounds with no improvement in the evaluation metric in order to stop the training.
//'   User can specify 0 to disable early stopping.
//' @param maximize whether to maximize the evaluation metric.
//' @param verbose logical value. Indicating if the validation result for each epoch is displayed or not.
//' @return a FTRL-Proximal linear model object
//' @export
// [[Rcpp::export]]
List FTRLProx_validate_spMatrix(S4 x, arma::vec y, const std::string family, List params, int epoch,
                                arma::sp_mat val_x, NumericVector val_y, Function eval,
                                int patience, bool maximize, bool verbose) {
  // Hyperparameter
  double alpha = as<double>(params["alpha"]);
  double beta = as<double>(params["beta"]);
  double l1 = as<double>(params["l1"]);
  double l2 = as<double>(params["l2"]);
  // Design Matrix
  arma::vec x_Dim = x.slot("Dim");
  arma::vec x_p = x.slot("p");
  arma::uvec x_i = x.slot("i");
  arma::vec x_x = x.slot("x");
  //Model Initialization
  arma::vec z(x_Dim[0], fill::zeros);
  arma::vec n(x_Dim[0], fill::zeros);
  arma::vec w(x_Dim[0], fill::zeros);
  // Model Prediction
  arma::vec p(x_Dim[1], fill::zeros);
  // Training and Validation Performance
  arma::vec eval_train(epoch, fill::zeros);
  arma::vec eval_val(epoch, fill::zeros);
  // Non-Zero Feature Count for in spMatrix
  arma::vec non_zero_count = diff(x_p);
  // Counter of Number of Rounds
  int Round;
  // Model Updating
  for (int r = 0; r < epoch; r++) {
    for (int t = 0; t < y.size(); t++) {
      // Non-Zero Feature Index in spMatrix
      arma::uvec non_zero_index = regspace<arma::uvec>(x_p[t], 1, x_p[t] + non_zero_count[t] - 1);
      // Non-Zero Feature Index for each sample
      arma::uvec i = x_i.elem(non_zero_index);
      // Non-Zero Feature Value for each sample
      arma::vec x_t_i = x_x.elem(non_zero_index);
      // Model Parameter
      arma::vec z_i = z.elem(i);
      arma::vec n_i = n.elem(i);
      // Computing Weight and Prediction
      arma::vec w_i = Weight_Update(alpha, beta, l1, l2, z_i, n_i);
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
    }
    eval_train[r] = as<double>(eval(p, y));
    eval_val[r] = as<double>(eval(FTRLProx_predict_spMatrix(val_x, w, family), val_y));
    Round = r;
    if (verbose == true) {
      Rcout << "[" << r+1 << "]"<< " \t train: " << eval_train[r] << " \t validation: " << eval_val[r] << std::endl;
    }
    if (patience != 0 & r + 1 >= patience) {
      if (maximize == true) {
        int round_max = as<int>(wrap(eval_val.index_max()));
        if (round_max <= r - patience) break;
      } else {
        int round_min = as<int>(wrap(eval_val.index_min()));
        if (round_min <= r - patience) break;
      }
    }
  }
  // Subset Evaluation Result to Performance
  arma::vec perf_train = eval_train.subvec(0, Round);
  arma::vec perf_val = eval_val.subvec(0, Round);
  // Retrun FTRL Proximal Model Weight and Performance
  return List::create(Named("weight") = w,
                      Named("perf_train") = perf_train,
                      Named("perf_val") = perf_val);
}
