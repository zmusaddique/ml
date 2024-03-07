import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Evaluatinon(ABC):
  """
  Abstract class for defining strategy for evaluation of our models
  """
  @abstractmethod
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    pass

class MSE(Evaluatinon):
  """
  Evaluation strategy that uses Mean Squared Error
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculating MSE...")
      mse = mean_squared_error(y_true, y_pred)
      logging.info(f"MSE: {mse}")
      return mse
    except Exception as e:
        logging.error(f"Error in calculating MSE: {e}")
        raise e
    
class R2(Evaluatinon):
  """
  Evaluation strategy that uses R2 score
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculating R2 score...")
      r2 = r2_score(y_true, y_pred)
      logging.info(f"R2: {r2}")
      return r2
    except Exception as e:
        logging.error(f"Error in calculating R2 score: {e}")
        raise e

class RMSE(Evaluatinon):
  """
  Evaluation strategy that uses Root Mean Squared Error
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculating RMSE...")
      mse = mean_squared_error(y_true, y_pred, squared=False)
      logging.info(f"MSE: {mse}")
      return mse
    except Exception as e:
        logging.error(f"Error in calculating RMSE: {e}")
        raise e