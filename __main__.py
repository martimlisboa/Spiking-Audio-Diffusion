from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train, train_distributed

from parser import make_parser

def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  replica_count = device_count()
  if replica_count > 1:
    if args.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {args.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    args.batch_size = args.batch_size // replica_count 
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args), nprocs=replica_count, join=True)
  else:
    train(args)


if __name__ == '__main__':
  main(make_parser().parse_args())
  
