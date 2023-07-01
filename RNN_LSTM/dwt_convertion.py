import pandas as pd
import numpy as np
from pywt import wavedec, coeffs_to_array
import os
from tqdm import tqdm

#fetch process, voltage and temperature from file name
def fetch_pvt_from_path(file_path):
    
    pvt = file_path.split("/")[-1][:-4].split("_")
    return pvt[0],float(pvt[1][:-1]),float(pvt[2])


#one hot encode for process
def one_hot_encode(p):
    
    p_list = ["fastnfastp","fastnslowp","slownfastp","slownslowp","typical"]
    encode = [0,0,0,0,0]
    
    for i in range(len(p_list)):
        if(p_list[i] == p):
            encode[i] = 1
            return encode
        
    raise Exception("encode problem") 


def convert_dwt_and_save(params,tort):
    input_folder_path = f"../../data/{tort}/"

    input_path = f"dwt_data/{tort}_input.csv"
    output_path = f"dwt_data/{tort}_output.csv"
    
    mode = params.mode
    level = params.level
    #find all the files in given path
    files = os.listdir(input_folder_path)

    input = []
    output = []

    #find dwt coefficent for all files
    for file in tqdm(files):
      file_path = f"{input_folder_path}{file}"

      #read csv 
      df = pd.read_csv(file_path)

      #pvt
      p,v,t = fetch_pvt_from_path(file_path) 
      pvt = [t]
      pvt.extend(one_hot_encode(p))   #[t,0,0,0,0,1] 

      vinn = df["vinn"] #output
      
      xpd = df["xpd"]   #input
      vdd = df["vdd"]
      vinp = df["vinp"]

      wave_size = len(vinn)

      #dwt convertion
      coeffs_vinn = wavedec(vinn.tolist(), 'db4', mode=mode,level=level)
      coeffs_xpd = wavedec(xpd.tolist(), 'db4', mode=mode,level=level)
      coeffs_vdd = wavedec(vdd.tolist(), 'db4', mode=mode,level=level)
      coeffs_vinp = wavedec(vinp.tolist(), 'db4', mode=mode,level=level)

      #flatten the coeffs
      flatten_output = coeffs_vinn[0]
      flatten_input = []

      for j in range(len(coeffs_xpd[0])):
        flatten_input.append(float(coeffs_xpd[0][j]))
        flatten_input.append(float(coeffs_vdd[0][j]))
        flatten_input.append(float(coeffs_vinp[0][j]))
        flatten_input.extend(pvt)
            
                
      output.append(flatten_output)
      input.append(flatten_input)


    #store coefficients in csv file
    np_input = np.array(input)
    df_input = pd.DataFrame(np_input)
    df_input.to_csv(input_path)

    np_output = np.array(output)
    df_output = pd.DataFrame(np_output)
    df_output.to_csv(output_path)

    print("dwt convertion completed...")

    return 
