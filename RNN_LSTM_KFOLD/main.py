from dwt_convertion import convert_dwt_and_save
from Parameters import Parameters
from train import build_train_and_save_model
from test import test_and_save
from plt import plot_result
from idwt import idwt_conversion

from create_folds import create_folds



params = Parameters(input_dim = 9, hidden_dim = 64,  batch_size=10, epochs = 2,learning_rate = 0.001,rnn="lstm",level = 4, mode="smooth",folds=5)
# params = Parameters(input_dim = 9, hidden_dim = 64,  batch_size=9 , epochs = 1,learning_rate = 0.001,rnn="lstm",level = 4, mode="smooth",folds=10)

# convert_dwt_and_save(params)

# create_folds(params)


for i in range(1,params.folds+1):   # we consider the ith fold for test and use remaining for testing
  print("epochs loop ",i)
  print("------------------------------------------------------------------------------------------------------------------")
  params.for_folds(i)
  # build_train_and_save_model(params,(i))

  test_and_save(params,i)
  idwt_conversion(params,i)
  plot_result(params,i)
