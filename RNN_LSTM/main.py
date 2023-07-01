from dwt_convertion import convert_dwt_and_save
from Parameters import Parameters
from train import build_train_and_save_model
from test import test_and_save
from plt import plot_result
from idwt import idwt_conversion



params = Parameters(input_dim = 9, hidden_dim = 64,  batch_size=10, epochs = 10,learning_rate = 0.001,rnn="lstm",level = 4, mode="smooth")

# convert_dwt_and_save(params,"train")
# convert_dwt_and_save(params,"test")

# build_train_and_save_model(params)

# test_and_save(params)

# idwt_conversion(params)

plot_result(params)
