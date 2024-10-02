#!/usr/bin/env python3

import os
import time
import copy
import pickle
import argparse
from itertools import product
from torch.utils.data import DataLoader
import torch
from prune.utils import *
from prune.Layer_wise_pruner import LayerPruner

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet20')
    parser.add_argument('--dset', type=str, default='cifar10')
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument('--shuffle_train', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--test_batch_size', type=int, default=500)
    parser.add_argument('--sample_size', type=int, default=128)
    parser.add_argument('--unstr', type=int, default=0)
    parser.add_argument('--seed', type=int, nargs='+', default=[42])
    parser.add_argument('--sparsity', type=float, nargs='+', default=[0.95])
    parser.add_argument('--lambda_inv', type=float, nargs='+', default=[0.0001])
    parser.add_argument('--algo', type=str, nargs='+', default=['Newton'])
    parser.add_argument('--lowr', type=float, nargs='+', default=[0.1])
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--nonuniform', action='store_true')
    parser.add_argument('--NM_N', type=str, default='1')
    parser.add_argument('--NM_M', type=str, default='4')
    parser.add_argument('--mask_alg', type=str, default='MP', help='Algorithm for mask selection (e.g., MP, L1)')
    parser.add_argument('--max_CG_iterations', type=int, default=1000, help='Maximum number of CG iterations')

    args = parser.parse_args()

    # Set up paths and device
    os.environ['IMAGENET_PATH'] = './datasets/MiniImageNet/imagenet-mini'
    dset_paths = {
        'imagenet': os.environ['IMAGENET_PATH'],
        'cifar10': './datasets',
        'cifar100': './datasets',
        'mnist': './datasets'
    }
    dset_path = dset_paths[args.dset]
    ROOT_PATH = './Sparse_NN'
    FOLDER = f'{ROOT_PATH}/results/{args.arch}_{args.dset}_{args.exp_name}'
    os.makedirs(FOLDER, exist_ok=True)
    FILE = f'{FOLDER}/data{args.exp_id}_{int(time.time())}.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Model setup
    model, train_dataset, test_dataset, criterion, modules_to_prune = model_factory(args.arch, dset_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel with {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
        modules_to_prune = ["module." + x for x in modules_to_prune]

    model.to(device)
    model.eval()

    # Set parameters based on architecture
    if args.arch == "resnet50_imagenet":
        batch_size = 100
        batching = True
        k_step = 30
        args.sample_size = 3000
        elimination_fraction = 1
        w_warm = False

    elif args.arch in ["resnet50_cifar10", "resnet50_cifar100"]:
        batch_size = 128
        batching = True
        k_step = 100
        args.sample_size = 3000
        elimination_fraction = 0.5
        w_warm = False

    elif args.arch == "resnet20_cifar10":
        batch_size = 1024
        batching = True
        k_step = 40
        args.sample_size = 3000
        elimination_fraction = 0.25
        w_warm = False

    else:
        # Default settings for other architectures
        batch_size = 128
        batching = True
        k_step = 100
        args.sample_size = 3000
        elimination_fraction = 0.5
        w_warm = False

    # Pruning and training
    for seed, sparsity, lambda_inv, algo, lowr in product(
        args.seed,
        args.sparsity,
        args.lambda_inv,
        args.algo,
        args.lowr
    ):
        print(f'Running with seed: {seed}, sparsity: {sparsity}, lambda_inv: {lambda_inv}, algo: {algo}')

        set_seed(seed)
        model_pruned = copy.deepcopy(model)

        pruner = LayerPruner(
            model_pruned,
            modules_to_prune,
            train_dataset,
            train_dataloader,
            test_dataloader,
            args.sample_size,
            criterion,
            lambda_inv,
            seed,
            device,
            algo,
            lowr,
            args.nonuniform
        )
        pruner.scaled = True

        start_time = time.time()

        model_update, accuracies, losses, layer_times, layer_wise_loss, layer_wise_W, layer_wise_size = pruner.prune_NM(
            N=int(args.NM_N),
            M=int(args.NM_M),
            k_step=k_step,
            w_warm=w_warm,
            save_memory=False,
            batching=batching,
            batch_size=batch_size,
            elimination_fraction=elimination_fraction,
            max_CG_iterations=args.max_CG_iterations,
            mask_alg=args.mask_alg
        )

        # Save results
        results = {'accuracies': accuracies, 'losses': losses}
        result_filename = f"./Results/{args.arch}_{algo}_{args.mask_alg}_{int(args.NM_N)}:{int(args.NM_M)}_k={k_step}_results.pickle"

        total_time = time.time() - start_time  # End the overall timing

        # Save the total time taken in the results
        results['total_time'] = total_time
        print(f"Total time for the entire process: {total_time:.2f} seconds")

        with open(result_filename, 'wb') as f:
            pickle.dump(results, f)

        # Save the pruned model
        model_filename = f"./Pruned_models/{args.arch}_{algo}_{args.mask_alg}_{int(args.NM_N)}:{int(args.NM_M)}_k={k_step}_model.pth"
        torch.save(model_update.state_dict(), model_filename)

        print(f"Results saved to {result_filename} and model saved to {model_filename} in {total_time:.2f} seconds")

        ls_filename = f"./Layer_wise/{args.arch}_{algo}_{args.mask_alg}_{int(args.NM_N)}:{int(args.NM_M)}_k={k_step}_layer_wise_data.pickle"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(ls_filename), exist_ok=True)

        # Create a dictionary containing all the layer-wise dictionaries
        layer_wise_data = {
            'layer_wise_losses': layer_wise_loss,
            'layer_wise_W': layer_wise_W,
            'layer_wise_size': layer_wise_size
        }

        # Save the combined dictionary as a pickle file
        with open(ls_filename, 'wb') as f:
            pickle.dump(layer_wise_data, f)

if __name__ == '__main__':
    main()
