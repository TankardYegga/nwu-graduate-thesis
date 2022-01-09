
import torch
from models.pinsage_pytorch.pinsage_parser import parse_args
import os
import pickle
from utils.load_data import Data

args = parse_args()
print('path is', args.data_path)
print('Total Epochs is', args.epoch)

data_generator = Data(path=args.data_path)
print('The data has been obtained!')
# print('n symptoms', data_generator.symptom_list)
# print('type', type(data_generator.symptom_list[0]))


# print('n herbs', data_generator.herb_list)
data_f_path = "datasets/data_generator.pkl"
try:
    with open(data_f_path, "rb") as f:
        data = pickle.load(f)
    print('data is', data)
    print(data.symptom_list)
except Exception:
    with open(data_f_path, "wb") as f:
        pickle.dump(data_generator, f, -1)




