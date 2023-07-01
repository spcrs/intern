import matplotlib.pyplot as plt
from pywt import waverec, wavedec, coeffs_to_array, array_to_coeffs
import pandas as pd
import numpy as np
import os
import math
import csv
from sklearn.metrics import r2_score
import math


def expected_out():
    #find all the files in given path
    files = os.listdir("data/train/")

    expected = []
    #find dwt coefficent for all files
    for file in (files):
      file_path = f"data/test/{file}"

      #read csv 
      df = pd.read_csv(file_path)
      vinn = list(df["vinn"])

      vinn.append(file)

      expected.append(vinn)

    return expected
    
     
def plot_result(params,fold):
  
  #prepare predicted data
  df = pd.read_csv(f"result/predicted_output_fold{fold}.csv")
  predicted = df.values[:,1:].tolist()

  expected = pd.read_csv(f"expected/expected_output_fold{fold}.csv").values[:,1:-1]

  file_detail = pd.read_csv(f"expected/expected_output_fold{fold}.csv").values[:,-1]



  smse = 0
  ssnr = 0
  sr2s = 0
  smae = 0
  
  data = []


  for i in range(len(expected)):
    
    mse = RMSE(expected[i][:params.wave_size],predicted[i][:params.wave_size])
    snr = SNR(expected[i][:params.wave_size],predicted[i][:params.wave_size])
    r2s = R2SCORE(expected[i][:params.wave_size],predicted[i][:params.wave_size])
    mae = MAE(expected[i][:params.wave_size],predicted[i][:params.wave_size])


    # data.append([file_detail[i],mse,snr,r2s,mae])
    smse+= mse
    ssnr += snr
    sr2s += r2s
    smae += mae
  
    
    plt.figure()
    plt.title(f" RMSE : {mse}  SNR : {snr} R2Score : {r2s} MAE L {mae}")
    plt.grid(True)
    plt.xlabel("Index", fontsize=10)
    

    plt.plot(range(len(expected[i][:params.wave_size])), expected[i][:params.wave_size], color="red", linewidth = 2, label = "actual vinn")
    plt.plot(range(len(predicted[i][:params.wave_size])), predicted[i][:params.wave_size], color="blue", linewidth = 1, label = "predicted vinn")

    plt.legend(['actual vinn', 'predicted vinn'])
    plt.savefig(f'graph/output/preds_folds{fold}{i}.png')



  no_of_test_files = len(os.listdir("../../data/train/"))/params.folds
  print("MSE : ",smse/no_of_test_files)
  print("SNR : ",ssnr/no_of_test_files)
  print("R2Score : ",sr2s/no_of_test_files)

  data = [fold,smse/no_of_test_files,ssnr/no_of_test_files,sr2s/no_of_test_files,smae/no_of_test_files]
  # Specify the filename
  filename = "result/Metrics.csv"
  if(fold == 1):
    # Write data to the CSV file
    with open(filename, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["folds,RMSE,SNR,R2Score,MAE"])
      writer.writerow(data)
  else:
    with open(filename, 'a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(data)
  # plt.show()


def SNR(y,y_pred):
  n = len(y)
  upper = 0
  lower = 0
  for ind in range(0,n):
      out = y[ind]
      pred_out = y_pred[ind]
      upper = upper + (out * out)
      lower = lower + (out - pred_out) * (out - pred_out)
  snr = 10 * math.log10(upper / (lower))
  return snr

def RMSE(y,y_pred):
  n = len(y)
  mse = 0
  for ind in range(0,n):
    mse += (y[ind]-y_pred[ind])**2
  return math.sqrt(mse/n)


def R2SCORE(y,y_pred):
   return r2_score(y, y_pred)

def MAE(y,y_pred):
  n = len(y)
  mae = 0
  for ind in range(0,n):
    mae += abs(y[ind]-y_pred[ind])
  return mae/n