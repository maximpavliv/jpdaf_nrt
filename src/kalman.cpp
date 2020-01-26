#include "kalman.h"

using namespace JPDAFTracker;
using namespace std;

Kalman::Kalman(const float& dt, const float& x, const float& y, const float& vx, const float& vy, const Eigen::Matrix2f& _R, const Eigen::Matrix2f& _T, const Eigen::Matrix4f& P_init)
{
  //TRANSITION MATRIX
  A << 1, dt, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, dt,
       0, 0, 0, 1;
  
	  
  //STATE OBSERVATION MATRIX
  C = Eigen::MatrixXf(2, 4);
  C << 1, 0, 0, 0,
       0, 0, 1, 0;
  
  //INITIAL COVARIANCE MATRIX
  P = P_init;
  //GAIN     
  K = Eigen::MatrixXf(4, 2);
  
  //measurement noise covariance matrix
  R = _R;

  T = _T;

  G = Eigen::MatrixXf(4, 2);
  G << std::pow(dt, 2) / 2, 0,
        	dt, 0,
        	0, std::pow(dt, 2) / 2,
        	0, dt;

  Q = G * T * G.transpose(); // TODO Check why and how



       
  last_prediction = cv::Point2f(x, y);
  last_speed = cv::Point2f(vx, vy);
  first = true;
}


cv::Point2f Kalman::predict()
{
  
  if(first)
  {
    x_predict << last_prediction.x, last_speed.x, last_prediction.y, last_speed.y;
    first = false;
  }
  else
  {
    x_predict = A*x_filter;
  }
  
  //Covariance Matrix predicted error
  P_predict = A * P * A.transpose() + Q;
  
  
  //Predicted Measurement
  z_predict = C * x_predict;
  
  //Error Measurement Covariance Matrix
  S = C * P_predict * C.transpose() + R;
  
  //Compute Entropy
//  entropy = k + .5 * log10(P.determinant());
  
  last_prediction_eigen = z_predict;
  last_prediction = cv::Point2f(z_predict(0), z_predict(1)); 
  return last_prediction;
}

void Kalman::gainUpdate(const float& beta)
{
  K = P_predict * C.transpose() * S.inverse();
  P = P_predict - (1 - beta) * K * S * K.transpose(); //TODO What?
}

Eigen::Vector4f Kalman::update(const std::vector< Eigen::Vector2f >& selected_detections, const Eigen::VectorXf& beta, const float& last_beta) //TODO beta seems to be some pounding coeff for selected detections. Check how computed! I hope sum(a)beta = 1. Seems like beta.size() = selected_detections.size() + 1, what is the last value then??. TODO But what is last_beta?? It is not even a pounded mean!! last_beta should be 0!! if not there is a big problem
{
  Eigen::Vector4f a;
  a.setZero();

  x_filter.setZero();
  uint i = 0;
  
  for(const auto& det : selected_detections)
  {
    a = x_predict + K * (det - z_predict);
    x_filter = x_filter + beta(i) * a;
    ++i;
  }
  
  x_filter = last_beta * x_predict + x_filter; //TODO Here check last_beta!
  
  a.setZero();
  Eigen::Matrix4f Pk = Eigen::Matrix4f::Zero();
  Eigen::Vector2f det;
  
  for(i = 0; i < selected_detections.size() + 1; ++i)
  {
    if(i == selected_detections.size())
    {
      a = x_predict;
    }
    else
    {
      a = x_predict + K * (selected_detections.at(i) - z_predict);
    }
    
    Pk = Pk + beta(i) * (a * a.transpose() - x_filter * x_filter.transpose());// TODO What??
    
  }
  
  P += Pk;

  return x_filter;
}



