from pywt import wavedec
import pandas as pd
from model_class.rnn_model import RNNModel
from model_class.lstm_model import LSTMModel
import os
class Parameters:
    def __init__(self,input_dim,hidden_dim,batch_size,epochs,learning_rate,rnn,level,mode):
      self.hidden_dim =  hidden_dim
      self.batch_size = batch_size
      self.epochs = epochs
      self.input_dim = input_dim
      self.learning_rate = learning_rate
      self.level = level
      self.mode = mode

      #to find size of wave
      example_file = "../../data/train/" + os.listdir("../../data/train/")[0]
      file = pd.read_csv(example_file)
      self.wave_size = len(file['vinn'])

      #to find no_of_coeffs to consider for model based on level and mode it will vary(approximation coefficient of last level is considered here)
      dummy_vinn = [i for i in range(0, self.wave_size)]
      dwt_coeffs = wavedec(dummy_vinn, 'db4', mode=mode,level=level)
      self.no_of_coeffs = len(dwt_coeffs[0])

      self.output_dim = self.no_of_coeffs
      self.save_path = f"models/{rnn}_{level}_{hidden_dim}_{epochs}_{learning_rate}_model.pt"

      if(rnn=="rnn"):
        self.model = RNNModel
      else:
        self.model = LSTMModel



