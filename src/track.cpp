#include "track.h"

using namespace JPDAFTracker;

Track::Track(const float& dt, const float& x, const float& y, const float& vx, const float& vy,
    const float& g_sigma, const float& gamma, const Eigen::Matrix2f& _R, const Eigen::Matrix4f& _Q, const Eigen::Matrix4f& P_init, int maxMissed, int minAcceptance)
  : g_sigma(g_sigma), gamma(gamma)
{
  KF = std::shared_ptr<Kalman>(new Kalman(dt, x, y, vx, vy, _R, _Q, P_init));
  life_time = 0;
  nodetections = 0;
  maxMissedRate = maxMissed;
  minAcceptanceRate = minAcceptance;
  entropy_sentinel = TrackState::NONE;
  id = -1;
}


cv::Point2f Track::predict()
{
  last_prediction = KF->predict();
  const Eigen::Matrix2f& S = KF->getS();
//  if(life_time == 0) 
//  {
//    initial_entropy = KF->getEntropy();
//  }
//  else if(nodetections >= maxNotDetection)
  if(nodetections >= maxMissedRate)
  {
    entropy_sentinel = TrackState::DISCARD;
  }
  else if(life_time >= minAcceptanceRate)
  {
    entropy_sentinel = TrackState::ACCEPT;
  }
  
  //Compute the volume VG
  ellipse_volume = CV_PI * g_sigma * sqrt(S.determinant());
  const float& param = ellipse_volume*gamma+1;
  number_returns = std::floor(param);
  side = sqrt(param / gamma) * .5; 
  return last_prediction;
}


