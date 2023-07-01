import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score
import csv
import matplotlib.pyplot as plt
import os

def prepare_predicted(fold,total_folds):
  no_of_test_files = 150//total_folds
  whole_predicted = [[] for i in range(no_of_test_files)]
  for split in range(1,6):
    df = pd.read_csv(f"result/result{split}/predicted_output_fold{fold}.csv")
    predicted = df.values[:,1:].tolist()
    for i in range(len(predicted)):
      whole_predicted[i].extend(predicted[i])

  return whole_predicted


def expected_out(fold,total_folds):
  no_of_test_files = 150//total_folds
  whole_expected = [[] for i in range(no_of_test_files)]
  for split in range(1,6):
    expected = pd.read_csv(f"expected/expected{split}/expected_output_fold{fold}.csv").values[:,1:]
    for i in range(len(expected)):
      whole_expected[i].extend(expected[i][:-1])
    
  return whole_expected




def combined(fold,total_folds):
  predicted = prepare_predicted(fold,total_folds)
  expected = expected_out(fold,total_folds)
  smse = 0
  ssnr = 0
  sr2s = 0
  smae = 0
  
  data = []


  for i in range(len(expected)):
    
    mse = RMSE(expected[i][:len(expected[i])],predicted[i][:len(expected[i])])
    snr = SNR(expected[i][:len(expected[i])],predicted[i][:len(expected[i])])
    r2s = R2SCORE(expected[i][:len(expected[i])],predicted[i][:len(expected[i])])
    mae = MAE(expected[i][:len(expected[i])],predicted[i][:len(expected[i])])


    data.append([mse,snr,r2s,mae])
    smse+= mse
    ssnr += snr
    sr2s += r2s
    smae += mae
  
    
    plt.figure()
    plt.title(f" RMSE : {mse}  SNR : {snr} R2Score : {r2s}")
    plt.grid(True)
    plt.xlabel("Index", fontsize=10)
    

    plt.plot(range(len(expected[i][:len(expected[i])])), expected[i][:len(expected[i])], color="red", linewidth = 2, label = "actual vinn")
    plt.plot(range(len(predicted[i][:len(expected[i])])), predicted[i][:len(expected[i])], color="blue", linewidth = 1, label = "predicted vinn")

    plt.legend(['actual vinn', 'predicted vinn'])
    plt.savefig(f'graph/output_combined/preds{i}_fold{fold}.png')



  no_of_test_files = len(os.listdir(f"data/test1/"))
  print("MSE : ",smse/no_of_test_files)
  print("SNR : ",ssnr/no_of_test_files)
  print("R2Score : ",sr2s/no_of_test_files)


  # Specify the filename
  filename = f"result/Metrics_combined{fold}.csv"

  # Write data to the CSV file
  with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file_name,RMSE,SNR,R2Score,MAE"])
    writer.writerows(data)

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