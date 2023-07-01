import pandas as pd
import torch
import numpy as np
import torch.nn as nn

def test_and_save(params,fold):
    
    input_coeffs_file = f"folds/test_input_fold{fold}.csv"
    output_coeffs_file = f"folds/test_output_fold{fold}.csv"
    save_output = f"result/predicted_coeffs_fold{fold}.csv"

    Model = params.model
    input_dim = params.input_dim
    hidden_dim = params.hidden_dim
    output_dim = params.output_dim
    batch_size = params.batch_size
    epochs = params.epochs
    no_of_coeffs = params.no_of_coeffs
    saved_model = params.save_path

    #prepare data
    input_df = pd.read_csv(input_coeffs_file)
    input_torch = (torch.tensor(input_df.values[:, 1:no_of_coeffs*input_dim+1], dtype=torch.float32)).reshape(-1, no_of_coeffs,input_dim)
    
    output_df = pd.read_csv(output_coeffs_file)
    expected_output = torch.tensor(output_df.values[:, 1:no_of_coeffs+1], dtype=torch.float)

    #restore the model
    model = Model(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(saved_model))
    model.eval()

    #test the model 
    with torch.no_grad():
      predicted_output = model(input_torch)
      predicted_output = predicted_output.reshape(-1,no_of_coeffs)
    loss_fn = nn.MSELoss()
    loss = loss_fn(predicted_output, expected_output)

    print("Testing Loss " ,loss)

    #save the output
    df_vinn = pd.DataFrame(np.array(predicted_output))
    df_vinn.to_csv(save_output)
