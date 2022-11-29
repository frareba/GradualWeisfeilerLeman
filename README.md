# Gradual Weisfeiler-Leman
Source code for the paper: F. Bause and N.M. Kriege, Gradual Weisfeiler-Leman: Slow and Steady Wins the Race. Proceedings of the First Learning on Graphs Conference (LoG 2022), PMLR 198, Virtual Event, December 9–12, 2022.

## Usage

This project contains implementations of the
- **gradual Weisfeiler-Leman subtree kernel** (gwlk)
- **gradual Weisfeiler-Leman optimal assignment kernel** (gwloa)
- Weisfeiler-Leman subtree kernel (wls)
- Weisfeiler-Leman optimal assignment kernel (wloa)

Use *./kkernel* to compute the corresponding kernel:
```
  Usage: kkernel [options] [command] [command options]
  Options:
    -a, --all
      Compute kernel for all data sets in the data directory
      Default: false
    -d, --datasets
      List of data sets
  Commands:
    wls      Compute the Weisfeiler-Lehman subtree kernel.
      Usage: wls [options]
        Options:
          -h, --height
            height, i.e., the number of refinement steps
            Default: [0, 1, 2, 3, 4, 5]

    wloa      Compute the Weisfeiler-Lehman optimal assignment kernel.
      Usage: wloa [options]
        Options:
          -h, --height
            height, i.e., the number of refinement steps
            Default: [0, 1, 2, 3, 4, 5]

    gwlk      Compute the gradual Weisfeiler-Lehman kernel using k-means 
            clustering. 
      Usage: gwlk [options]
        Options:
          -h, --height
            height, i.e., the number of refinement steps
            Default: [0, 1, 2, 3, 4, 5]
          -k, --k
            k of kmeans, i.e., the number of clusters
            Default: 4

    gwloa      Compute the gradual Weisfeiler-Lehman optimal assignment 
            kernel. 
      Usage: gwloa [options]
        Options:
          -h, --height
            height, i.e., the number of refinement steps
            Default: [0, 1, 2, 3, 4, 5]
          -k, --k
            k of kmeans, i.e., the number of clusters
            Default: 4
```

Normalize the gram matrices if wanted (*./kgram* norm -a)

Run *./kacc* for evaluation. Results are saved to log_runtime.txt, log_accuracy.txt and log_selection.txt.

A thorough description of these methods can be found in our paper and the corresponding references.  

### Examples
```
./kkernel -a gwlk -k 16
./kgram norm -a
./kacc
```
    
## Datasets
The repository contains the dataset *PTC_FM* only. Further datasets in the required format are available from the website [TUDatasets: A collection of benchmark datasets for graph classification and regression](https://chrsmrrs.github.io/datasets/).

The code for generating synthetic datasets can be found [here](https://github.com/mlai-bonn/BlockGraphGenerator).


## Terms and conditions
When using our code please cite our paper [Gradual Weisfeiler-Leman: Slow and Steady Wins the Race](https://arxiv.org/abs/2209.09048):

F. Bause and N.M. Kriege, Gradual Weisfeiler-Leman: Slow and Steady Wins the Race. Proceedings of the First Learning on Graphs Conference (LoG 2022), PMLR 198, Virtual Event, December 9–12, 2022.


## Contact information
If you have any questions, please contact [Franka Bause](https://dm.cs.univie.ac.at/team/person/112939/).
