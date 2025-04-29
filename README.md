# A hybrid perceptron with cross-domain transferability towards active steady-state non-line-of-sight imaging
## Introduction
This is an official PyTorch implementation of **"A hybrid perceptron with cross-domain transferability towards active steady-state non-line-of-sight imaging"**. Datasets and source code are available here. If you find this work is useful, please give it a star ⭐ and consider citing this paper in your research. Thank you!

## Dependencies and Installation

To set up the environment, please make sure you have Python 3.7 (or higher) and the following dependencies installed:

- **Python 3.7** or higher
- **Pytorch 1.8** or higher

## Datasets
Datasets used in this work include both synthetic and real-world data. The rendering pipeline for the synthetic dataset is based on [1]. The real-world dataset contains 4,620 non-line-of-sight (NLOS) image pairs captured from real-world scenes. The data includes 660 digits (ranging from 0 to 9) from the MNIST handwritten digit database, all manufactured using 3D printing technology. Out of these, 4,200 pairs are used for training/validation, and 420 pairs are used for testing/evaluation. Download the real-world dataset here [https://github.com/RL-NLOS/NLOS_HP-CDT/releases].

## References
[1] W. Chen, S. Daneau, F. Mannan, F. Heide, Steady-state non-line-of-sight imaging, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 6783–6792. doi:10.1109/CVPR.2019.00695.
