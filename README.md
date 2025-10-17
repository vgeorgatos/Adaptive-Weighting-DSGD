# Adaptive-Weighting-DSGD
This repository is the official code release for the methods presented in the [Fully Distributed Federated Learning with Efficient Local Cooperations](https://ieeexplore.ieee.org/abstract/document/10095741) (ICASSP '23) and its extended journal version (under submission, to be announced soon).

## Abstract

Recently, a shift has been observed towards the so-called edge machine learning, which allow multiple devices with local computational and storage resources to collaborate with the assistance of a centralized server. The well-known federated learning approach is able to utilize such architectures by allowing the exchange of only parameters with the server, while keeping the datasets private to each contributing device. In this work, we propose a communication-efficient, fully distributed, diffusion-based learning algorithm that does not require a parameter server and propose an adaptive combination rule for the cooperation of the devices. By adopting a classification task on the MNIST dataset, the efficacy of the proposed algorithm is demonstrated in non-IID dataset scenarios.

## Usage


Methods are implemented as Jupyter notebooks (.ipynb) for easy usage and are based on the PyTorch library. Codes are implmented to run on a single GPU.
Notebooks can be also easily converted to .py python files for command line execution ( https://mljar.com/blog/convert-jupyter-notebook-python/).

## Libraries


Python 3.11.5

PyTorch 2.6.0+cu118

FedLab 1.3.0

## References


If you found useful these  codes please cite the following papers:

```latex
@INPROCEEDINGS{10095741,
  author={Georgatos, Evangelos and Mavrokefalidis, Christos and Berberidis, Kostas},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Fully Distributed Federated Learning with Efficient Local Cooperations}, 
  year={2023},
  doi={10.1109/ICASSP49357.2023.10095741}}
```
Journal version 
(to be announced soon)


