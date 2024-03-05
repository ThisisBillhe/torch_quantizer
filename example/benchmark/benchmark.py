import argparse
import torch_quantizer as tq

def main():
    parser = argparse.ArgumentParser(description='Benchmarking script for linear and conv2d operations.')
    
    # Common arguments
    parser.add_argument('--o', type=str, choices=['linear', 'conv2d'], help='The benchmark operation to perform: linear or conv2d')
    
    # Arguments for linear benchmark
    parser.add_argument('--bs', type=int, help='Batch size', default=1)
    parser.add_argument('--cin', type=int, help='Channels in', default=960)
    parser.add_argument('--cout', type=int, help='Channels out', default=960)
    
    # Additional arguments for conv2d benchmark
    parser.add_argument('--h', type=int, help='Height of the input image', default=64)
    parser.add_argument('--w', type=int, help='Width of the input image', default=64)
    parser.add_argument('--k', type=int, help='Kernel size', default=3)
    parser.add_argument('--p', type=int, help='Padding size', default=0)
    
    args = parser.parse_args()

    if args.o == 'linear':
        # Ensure required arguments for linear operation are provided
        if args.cin > 0 and args.cout > 0:
            # tq.benchmark_linear(bs=args.bs, cin=args.cin, cout=args.cout)
            # use qlinear inheritted from nn.Linear for benchmark
            tq.benchmark_linearInheritance(bs=args.bs, cin=args.cin, cout=args.cout)
        else:
            print("Error: cin and cout are required for linear benchmark.")
    
    elif args.o == 'conv2d':
        # Ensure required arguments for conv2d operation are provided
        if args.cin > 0 and args.cout > 0 and args.h > 0 and args.w > 0 and args.k > 0:
            tq.benchmark_conv2d(bs=args.bs, cin=args.cin, h=args.h, w=args.w, cout=args.cout, k=args.k, padding=args.p)
        else:
            print("Error: cin, cout, h, w, and k are required for conv2d benchmark.")

if __name__ == '__main__':
    main()
