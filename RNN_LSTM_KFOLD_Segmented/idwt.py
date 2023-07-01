from pywt import wavedec,coeffs_to_array,array_to_coeffs,waverec
import pandas as pd
import numpy as np
def idwt_conversion(params,fold,split):
  predicted_coeffs = f"result/result{split}/predicted_coeffs_fold{fold}.csv"
  df = pd.read_csv(predicted_coeffs)
  preds = df.values[:,1:].tolist()
  predicted = []
  for pred in preds:

    dummy_vinn = [i for i in range(0, params.wave_size)]
    coeffs_dummy_vinn = wavedec(dummy_vinn, 'db4', mode=params.mode,level=params.level)
    dwt_coeff_slices = coeffs_to_array(coeffs_dummy_vinn)[1]
    
    total_coeffs = dwt_coeff_slices[-1]['d'][0].stop
    for i in range(len(pred),total_coeffs):
      pred.append(0)

    res = array_to_coeffs(pred, dwt_coeff_slices, output_format='wavedec')
    pyarr = waverec(res, 'db4', mode=params.mode)

    predicted.append(pyarr.tolist())

    df_vinn = pd.DataFrame(np.array(predicted))
    df_vinn.to_csv(f"result/result{split}/predicted_output_fold{fold}.csv")