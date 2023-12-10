import sys
import argparse
import os

# Prevent Python from generating .pyc files
sys.dont_write_bytecode = True

def prepare_data(args):
    """
    Prepare data based on the specified data type (synthetic or real).

    Args:
    args (Namespace): Parsed command line arguments.
    """
    if args.data_type == 'synthetic':
        os.system(f'python prepare_synthetic_data.py --graph_directed {args.graph_directed} '
                  f'--dynamic_weights {args.dynamic_weights} --edge_percent_reduce {args.edge_percent_reduce} '
                  f'--request_total {args.synthetic_request_total} --seed {args.seed}')
        os.system(f'python Embeddings/create_embedding.py --data_type {args.data_type} --num_nodes {args.num_nodes} '
                  f'--temporal {args.temporal} --emb_style {args.emb_style} --directed {args.graph_directed} '
                  f'--emb_size {args.embedding_size} --dynamic_weights {args.dynamic_weights} '
                  f'--edge_percent_reduce {args.edge_percent_reduce} '
                  f'--request_total {args.synthetic_request_total if args.data_type == "synthetic" else args.real_request_total} '
                  f'--seed {args.seed}')
    else:
        os.system(f'python prepare_real_data.py --num_nodes {args.num_nodes} --request_total {args.real_request_total} '
                  f'--temporal {args.temporal} --seed {args.seed}')

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_type', '--data_type', choices=['real', 'synthetic'], default='synthetic')
    parser.add_argument('-emb_style', '--emb_style', type=str, choices=['N2V', 'NeurADP'], default='NeurADP')
    parser.add_argument('-graph_directed', '--graph_directed', type=int, default=1)
    parser.add_argument('-dynamic_weights', '--dynamic_weights', type=int, default=1)
    parser.add_argument('-edge_percent_reduce', '--edge_percent_reduce', type=float, default=0.3)
    parser.add_argument('-synthetic_request_total', '--synthetic_request_total', type=int, default=100)
    parser.add_argument('-real_request_total', '--real_request_total', type=int, default=100)
    parser.add_argument('-num_nodes', '--num_nodes', type=int, default=150)
    parser.add_argument('-temporal', '--temporal', type=int, default=1)
    parser.add_argument('-seed', '--seed', type=int, default=5)
    parser.add_argument('-embedding_size', '--embedding_size', type=int, default=100)
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    # Prepare data based on parsed arguments
    prepare_data(args)
