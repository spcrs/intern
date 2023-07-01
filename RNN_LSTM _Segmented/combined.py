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
  no_of_test_files = len(os.listdir(f"data/test{1}/"))
  whole_expected = [[] for i in range(no_of_test_files)]
  #find all the files in given path
  for split in range(1,6):
    files = os.listdir(f"data/test{split}/")

    expected = []
    #find dwt coefficent for all files
    file_details = []
    for file in (files):
      file_path = f"data/test{split}/{file}"

      #read csv 
      df = pd.read_csv(file_path)
      vinn = list(df["vinn"])

      file_details.append(file)

      expected.append(vinn)

    for i in range(len(expected)):
      whole_expected[i].extend(expected[i])
 
  return whole_expected,file_details

def prepare_predicted():
  no_of_test_files = len(os.listdir(f"data/test{1}/"))
  whole_predicted = [[] for i in range(no_of_test_files)]
  for split in range(1,6):
    df = pd.read_csv(f"result/predicted_output{split}.csv")
    predicted = df.values[:,1:].tolist()
    for i in range(len(predicted)):
      whole_predicted[i].extend(predicted[i])

  return whole_predicted
     
# def plot_result(params,split):
def combined():

  #prepare predicted data
  predicted = prepare_predicted()
  expected,file_detail = expected_out()


  # for i in expected:
  #   file_detail.append(i[-1])
  #   expected_output.append(i[:-1])

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


    data.append([file_detail[i],mse,snr,r2s,mae])
    smse+= mse
    ssnr += snr
    sr2s += r2s
    smae += mae
  
    
    plt.figure()
    plt.title(f"{file_detail[i]} RMSE : {mse}  SNR : {snr} R2Score : {r2s}")
    plt.grid(True)
    plt.xlabel("Index", fontsize=10)
    

    plt.plot(range(len(expected[i][:len(expected[i])])), expected[i][:len(expected[i])], color="red", linewidth = 2, label = "actual vinn")
    plt.plot(range(len(predicted[i][:len(expected[i])])), predicted[i][:len(expected[i])], color="blue", linewidth = 1, label = "predicted vinn")

    plt.legend(['actual vinn', 'predicted vinn'])
    plt.savefig(f'graph/output_combined/preds{i}.png')



  no_of_test_files = len(os.listdir(f"data/test1/"))
  print("MSE : ",smse/no_of_test_files)
  print("SNR : ",ssnr/no_of_test_files)
  print("R2Score : ",sr2s/no_of_test_files)


  # Specify the filename
  filename = f"result/Metrics_combined.csv"

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