from models.pinsage.pinsage import PinSage
import tensorflow as tf
from models.pinsage import *

if __name__ == '__main__':
    from utils.load_data import Data
    from models.pinsage.pinsage_parser import parse_args
    with tf.device('/cpu:0'):
        args = parse_args()

        print('path is', args.data_path)
        print('Total Epochs is', args.epoch)

        data_generator = Data(path=args.data_path, batch_size=args.batch_size)
        print('The data has been obtained!')

        model = PinSage(data_generator=data_generator, args=args)
        print(model)

    