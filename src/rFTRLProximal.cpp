// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include <RcppArmadillo.h>
#include <progress.hpp>
using namespace Rcpp;
using namespace arma;
using namespace std;

double PredTransform(double x, std::string family) {
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

NumericVector Weight_Update(double alpha, double beta, double l1, double l2, NumericVector z, NumericVector n) {
  NumericVector eta = (beta + sqrt(n)) / alpha + l2;
  NumericVector w = -1 / eta * (z - as<NumericVector>(sign(z)) * l1);
  w[abs(z) <= l1] = 0;
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
//' @param w an vector of linear model weights.
//' @param family link function to be used in the model. "gaussian", "binomial" and "poisson" are avaliable.
//' @return an vector of linear model predicted values
//' @export
// [[Rcpp::export]]
NumericVector FTRLProx_predict_spMatrix(arma::sp_mat x, NumericVector w, std::string family) {
  arma::vec p_vec = arma::vectorise(as<arma::rowvec>(w) * x);
  NumericVector p = NumericVector(p_vec.begin(), p_vec.end());
  if (family == "gaussian") {
    return p;
  } else if (family == "binomial"){
    return 1 / (1 + exp(-p));
  } else if (family == "poisson") {
    return exp(p);
  } else {
    return NumericVector(p.length());
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
//' @return an vector of linear model weights
//' @export
// [[Rcpp::export]]
NumericVector FTRLProx_train_spMatrix(arma::sp_mat x, NumericVector y, std::string family, List params, int epoch, bool verbose) {
  // Hyperparameter
  double alpha = as<double>(params["alpha"]);
  double beta = as<double>(params["beta"]);
  double l1 = as<double>(params["l1"]);
  double l2 = as<double>(params["l2"]);
  // Design Matrix
  IntegerVector x_Dim = IntegerVector::create(x.n_rows, x.n_cols);
  IntegerVector x_p(x.col_ptrs, x.col_ptrs + x.n_cols + 1);
  IntegerVector x_i(x.row_indices, x.row_indices + x.n_nonzero);
  NumericVector x_x(x.values, x.values + x.n_nonzero);
  //Model Initialization
  NumericVector z(x_Dim[0]);
  NumericVector n(x_Dim[0]);
  NumericVector w(x_Dim[0]);
  // Model Prediction
  NumericVector p(x_Dim[1]);
  // Computing
  // Non-Zero Feature Count for in spMatrix
  IntegerVector non_zero_count = diff(x_p);
  // Initialize Progress Bar
  if (verbose == true) {
    Function msg("message");
    msg(std::string("Model Computing:"));
  }
  Progress pb (epoch * x_Dim[1], verbose);
  // Model Updating
  for (int r = 0; r < epoch; r++) {
    if (Progress::check_abort()) {
      return w;
    }
    for (int t = 0; t < y.size(); t++) {
      // Non-Zero Feature Index in spMatrix
      IntegerVector non_zero_index = x_p[t] + seq_len(non_zero_count[t]) - 1;
      // Non-Zero Feature Index for each sample
      IntegerVector i = x_i[non_zero_index];
      // Non-Zero Feature Value for each sample
      NumericVector x_i = x_x[non_zero_index];
      // Model Parameter
      NumericVector z_i = z[i];
      NumericVector n_i = n[i];
      // Computing Weight and Prediction
      NumericVector w_i = Weight_Update(alpha, beta, l1, l2, z_i, n_i);
      double p_t = PredTransform(sum(x_i * w_i), family);
      // Updating Weight and Prediction
      w[i] = w_i;
      p[t] = p_t;
      // Computing Model Parameter of Next Round
      NumericVector g_i = (p[t] - y[t]) * x_i;
      NumericVector s_i = (sqrt(n_i + pow(g_i, 2)) - sqrt(n_i)) / alpha;
      NumericVector z_i_next = z_i + g_i - s_i * w_i;
      NumericVector n_i_next = n_i + pow(g_i, 2);
      // Updating Model Parameter
      z[i] = z_i_next;
      n[i] = n_i_next;
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
//' @param verbose logical value. Indicating if the validation result for each epoch is displayed or not.
//' @return a FTRL-Proximal linear model object
//' @export
// [[Rcpp::export]]
List FTRLProx_validate_spMatrix(arma::sp_mat x, NumericVector y, std::string family, List params, int epoch,
                                arma::sp_mat val_x, NumericVector val_y, Function eval, bool verbose) {
  // Hyperparameter
  double alpha = as<double>(params["alpha"]);
  double beta = as<double>(params["beta"]);
  double l1 = as<double>(params["l1"]);
  double l2 = as<double>(params["l2"]);
  // Design Matrix
  IntegerVector x_Dim = IntegerVector::create(x.n_rows, x.n_cols);
  IntegerVector x_p(x.col_ptrs, x.col_ptrs + x.n_cols + 1);
  IntegerVector x_i(x.row_indices, x.row_indices + x.n_nonzero);
  NumericVector x_x(x.values, x.values + x.n_nonzero);
  //Model Initialization
  NumericVector z(x_Dim[0]);
  NumericVector n(x_Dim[0]);
  NumericVector w(x_Dim[0]);
  // Model Prediction
  NumericVector p(x_Dim[1]);
  // Training and Validation Performance
  NumericVector eval_train(epoch);
  NumericVector eval_val(epoch);
  // Computing
  // Non-Zero Feature Count for in spMatrix
  IntegerVector non_zero_count = diff(x_p);
  // Model Updating
  for (int r = 0; r < epoch; r++) {
    for (int t = 0; t < y.size(); t++) {
      // Non-Zero Feature Index in spMatrix
      IntegerVector non_zero_index = x_p[t] + seq_len(non_zero_count[t]) - 1;
      // Non-Zero Feature Index for each sample
      IntegerVector i = x_i[non_zero_index];
      // Non-Zero Feature Value for each sample
      NumericVector x_i = x_x[non_zero_index];
      // Model Parameter
      NumericVector z_i = z[i];
      NumericVector n_i = n[i];
      // Computing Weight and Prediction
      NumericVector w_i = Weight_Update(alpha, beta, l1, l2, z_i, n_i);
      double p_t = PredTransform(sum(x_i * w_i), family);
      // Updating Weight and Prediction
      w[i] = w_i;
      p[t] = p_t;
      // Computing Model Parameter of Next Round
      NumericVector g_i = (p[t] - y[t]) * x_i;
      NumericVector s_i = (sqrt(n_i + pow(g_i, 2)) - sqrt(n_i)) / alpha;
      NumericVector z_i_next = z_i + g_i - s_i * w_i;
      NumericVector n_i_next = n_i + pow(g_i, 2);
      // Updating Model Parameter
      z[i] = z_i_next;
      n[i] = n_i_next;
    }
    eval_train[r] = as<double>(eval(p, y));
    eval_val[r] = as<double>(eval(FTRLProx_predict_spMatrix(val_x, w, family), val_y));
    if (verbose == true) {
      Rcout << "[" << r+1 << "]"<< " \t train: " << eval_train[r] << " \t validation: " << eval_val[r] << std::endl;
    }
  }
  // Retrun FTRL Proximal Model Weight and Performance
  return List::create(Named("weight") = w,
                      Named("eval_train") = eval_train,
                      Named("eval_val") = eval_val);
}
