from Training import train
import argparse


"""
main script

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Trains the ED network. Can also be used to run pure inference.')

    parser.add_argument('--model_save_dir', default='../Models/', type=str, nargs='?', help='Folder directory in which to store the trained models.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--datasets', default=" ", nargs='+',type=str, help='The names of the datasets to use.')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--batch_size', default=8, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--w_length', default=16, type=int, nargs='?', help='Input temporal size.')

    parser.add_argument('--units', default=8, nargs='+', type=int, help='Hidden layer sizes (amount of units) of the network.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    return parser.parse_args()


def start_train(args):

    if args.only_inference:
        if args.dataset == 'CL1B':
            cond = 4
            w_length = 32
        elif args.dataset == 'LA2A':
            cond = 2
            w_length = 32
        else:
            cond = 4
            w_length = 64

    print("######### Preparing for training/inference #########")
    print("\n")
    train(data_dir=args.data_dir,
          model_save_dir=args.model_save_dir,
          save_folder=f'{args.dataset} Model',
          dataset=args.datasets,
          epochs=args.epochs,
          cond=cond,
          w_length=w_length,
          b_size=args.batch_size,
          units=args.units,
          learning_rate=args.learning_rate,
          inference=args.only_inference)

def main():
    args = parse_args()
    start_train(args)

if __name__ == '__main__':
    main()