from pywt import wavedec
import pandas as pd
from model_class.rnn_model import RNNModel
from model_class.lstm_model import LSTMModel
import os
class Parameters:
    def __init__(self,input_dim,hidden_dim,batch_size,epochs,learning_rate,rnn,level,mode,folds,split):
      self.hidden_dim =  hidden_dim
      self.batch_size = batch_size
      self.epochs = epochs
      self.input_dim = input_dim
      self.learning_rate = learning_rate
      self.level = level
      self.mode = mode
      self.folds = folds
      self.rnn = rnn



      #to find size of wave
      example_file = f"data/train{split}/" + os.listdir(f"data/train{split}/")[0]
      file = pd.read_csv(example_file)
      self.wave_size = len(file['vinn'])

      #to find no_of_coeffs to consider for model based on level and mode it will vary(approximation coefficient of last level is considered here)
      dummy_vinn = [i for i in range(0, self.wave_size)]
      dwt_coeffs = wavedec(dummy_vinn, 'db4', mode=mode,level=level)
      self.no_of_coeffs = len(dwt_coeffs[0])

      self.output_dim = self.no_of_coeffs

      if(rnn=="rnn"):
        self.model = RNNModel
      else:
        self.model = LSTMModel


    def for_folds(self,fold,split):
      self.save_path = f"models/models{split}/{self.rnn}_{self.level}_{self.hidden_dim}_{self.epochs}_{self.learning_rate}_model_fold{fold}.pt"
