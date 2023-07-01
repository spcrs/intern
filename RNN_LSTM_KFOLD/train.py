import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def build_train_and_save_model(params,fold):


    # input_dim, hidden_dim, output_dim,batch_size, epochs,
    Model = params.model
    input_dim = params.input_dim
    hidden_dim = params.hidden_dim
    output_dim = params.output_dim
    batch_size = params.batch_size
    epochs = params.epochs
    no_of_coeffs = params.no_of_coeffs
    save_path = params.save_path

    print("build and train started")

    # prepare input and output data
    input_df = pd.DataFrame()
    output_df = pd.DataFrame()
    input_coeffs_file = f"folds/train_input_fold{fold}.csv"
    output_coeffs_file = f"folds/train_output_fold{fold}.csv"
    input_df = pd.read_csv(input_coeffs_file)
    output_df = pd.read_csv(output_coeffs_file)
       
    input_torch = (torch.tensor(input_df.values[:, 1:no_of_coeffs*input_dim+1], dtype=torch.float32)).reshape(-1, no_of_coeffs, input_dim)
    expected_output = torch.tensor(output_df.values[:, 1:no_of_coeffs+1], dtype=torch.float)
    print(input_torch.shape)
    print("files read successfully")

    # build model
    model = Model(input_dim, hidden_dim, output_dim)

    # create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = nn.MSELoss()

    # train the model
    train = TensorDataset(input_torch, expected_output)
    train_loader = DataLoader(train, batch_size, shuffle=True)

    losses = []
    total_epochs = epochs
    while(epochs != 0):
        for j, (inp, out) in enumerate(train_loader):

            model_output = model(inp.reshape(batch_size, no_of_coeffs,input_dim))
            loss = loss_fn(model_output, out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
        print(f"{epochs} - {loss}")
        losses.append(float(loss)) 
        
        epochs -=1 

    torch.save(model.state_dict(), save_path)
    print(f"Training loss {loss}")

    
    print("Total epochs is ", total_epochs)
    