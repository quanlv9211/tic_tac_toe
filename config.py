import argparse
from email.policy import default
import torch
import os

parser = argparse.ArgumentParser(description='Tic Tac Toe Config')

# init
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--width', type=int, default=3, help='chieu rong cua bang')
parser.add_argument('--height', type=int, default=3, help='chieu dai cua bang')
parser.add_argument('--winstreak', type=int, default=3, help='so o lien tiep de thang')
parser.add_argument('--your_model', type=str, default='random', help='mo hinh nguoi choi')
parser.add_argument('--opponent_model', type=str, default='random', help='mo hinh doi thu')
parser.add_argument('--your_symbol', type=str, default='X', help='ban chon X hay O, X di truoc, O di sau')
parser.add_argument('--opponent_symbol', type=str, default='O', help='doi thu chon X hay O, X di truoc, O di sau')

# training
parser.add_argument('--device', type=str, default='cpu', help='thiet bi huan luyen mo hinh')
parser.add_argument('--device_id', type=str, default='0', help='id gpu')
parser.add_argument('--epoch', type=int, default=1e5, help='so epoch huan luyen')
parser.add_argument('--lr', type=float, default=0.01, help='so o lien tiep de thang')
parser.add_argument('--log_interval', type=int, default=500, help='in ra thong bao sau bao nhieu epoch')
parser.add_argument('--test_games', type=int, default=100, help='so game kiem tra')
parser.add_argument('--render', type=bool, default=False, help='in ra ban co trong qua trinh choi khong')
parser.add_argument('--function', type=str, default='', help='train/test')

# specific argument
parser.add_argument('--minimax_depth', type=int, default=2, help='chieu sau cua thuat toan minimax_depth')

args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    print('using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

args.output_folder = '../output/log/opponent_{}/you_{}/'.format(args.opponent_model, args.your_model)
args.checkpoint_folder = '../checkpoint/opponent_{}/you_{}/'.format(args.opponent_model, args.your_model)