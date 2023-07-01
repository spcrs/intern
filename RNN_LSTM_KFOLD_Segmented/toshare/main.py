from dwt_convertion import convert_dwt_and_save
from Parameters import Parameters
from train import build_train_and_save_model
from test import test_and_save
from plt import plot_result
from idwt import idwt_conversion
from combined import combined
from create_folds import create_folds

hidden_dims = [64,64,64,64,64]
batch_sizes = [10,10,10,10,10]
epochs = [100,100,100,100,100]
learning_rates = [0.001,0.001,0.001,0.001,0.001]
rnns = ["lstm","lstm","lstm","lstm","lstm"]
levels = [4,3,5,6,3]
modes = ["smooth","smooth","smooth","smooth","smooth"]

total_folds = 5


# for split in range(1,6):
#   params = Parameters(input_dim = 9, hidden_dim = hidden_dims[split-1],  batch_size=batch_sizes[split-1], epochs = epochs[split-1],learning_rate = learning_rates[split-1],rnn=rnns[split-1],level = levels[split-1], mode=modes[split-1],folds = total_folds,split=split)
#   ##### convert_dwt_and_save(params,split)
#   ##### create_folds(params,split)

#   for i in range(1,params.folds+1):   # we consider the ith fold for test and use remaining for testing
#     print("epochs loop ",i)
#     print("------------------------------------------------------------------------------------------------------------------")
#     params.for_folds(i,split)
#     build_train_and_save_model(params,(i),split)

#     test_and_save(params,i,split)
#     idwt_conversion(params,i,split)
#     plot_result(params,i,split)

for fold in range(total_folds,total_folds+1):
  combined(fold,total_folds)