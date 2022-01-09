# if __name__ == "__main__":
#     from utils.parser import parse_args
#     from utils.load_data import Data

#     args = parse_args()

#     print('path is', args.data_path)
#     print('Total Epochs is', args.epoch)

#     data_generator = Data(path=args.data_path, batch_size=args.batch_size)
#     print('The data has been obtained!')

import scipy.sparse as sp
import numpy as np
sympt_herb_cocur_nums_mat = sp.load_npz("datasets/TCM1sympt_herb_cocur_nums_mat.npz")
