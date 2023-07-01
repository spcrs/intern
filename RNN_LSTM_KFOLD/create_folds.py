import pandas as pd
from sklearn.model_selection import KFold
import os


def expected_out():
    #find all the files in given path

    expected = []
    for tort in ["train","test"]:
        files = os.listdir(f"../../data/{tort}/")

        #find dwt coefficent for all files
        for file in (files):
            file_path = f"../../data/{tort}/{file}"

            #read csv 
            df = pd.read_csv(file_path)
            vinn = list(df["vinn"])
            if(len(vinn) % 2 == 1):
                vinn.append(vinn[-1])
            vinn.append(file)

            expected.append(vinn)
    print(len(expected))
    df_output = pd.DataFrame(expected)
    return df_output

def create_folds(params):
    # Load the dataset from CSV
    input_data = pd.read_csv('dwt_data/train_input.csv')
    output_data = pd.read_csv('dwt_data/train_output.csv')
    

    # Set the number of folds
    k = params.folds

    # Initialize the KFold object
    kf = KFold(n_splits=k,shuffle=True)  #check with stratified k fold   

    # Iterate over the folds
    expected = expected_out()
    for fold, (train_index, test_index) in enumerate(kf.split(input_data)):
        # Create the train and test indices for the current fold
        train_input_index = train_index
        test_input_index = test_index
        # print(input_data)
        curr = expected.iloc[test_index]
        curr.to_csv(f'expected/expected_output_fold{fold+1}.csv')

        # Use the same indices for the output data
        train_output_index = train_index
        test_output_index = test_index
        
        # Create the train and test datasets for the current fold
        train_input = input_data.iloc[train_input_index]
        test_input = input_data.iloc[test_input_index]
        
        train_output = output_data.iloc[train_output_index]
        test_output = output_data.iloc[test_output_index]
        
        # Save the train and test datasets to separate CSV files
        train_input.to_csv(f'folds/train_input_fold{fold + 1}.csv', index=False)
        train_output.to_csv(f'folds/train_output_fold{fold + 1}.csv', index=False)
        test_input.to_csv(f'folds/test_input_fold{fold + 1}.csv', index=False)
        test_output.to_csv(f'folds/test_output_fold{fold + 1}.csv', index=False)
