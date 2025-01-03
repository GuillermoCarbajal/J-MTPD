# J-MTPD

Official Pytorch Implementation  of *J-MTPD*. 
 
| **Blurry** | **Kernels** | **J-MTPD (optim)** |
|:------------:|:------------:|:------------:|
| ![Imagen 1](testing_imgs/Blurry2_8_img_blurry.png) | ![Imagen 2](testing_imgs/Blurry2_8_kernels.png) | ![Imagen 3](testing_imgs/Blurry2_8_img_restored.png) |

## Trajectory Prediction Network Architecture
<p align="center">
<img width="900" src="imgs/two_branches.png?raw=true">
  </p>
  
## Quick Demo


* <a href="https://colab.research.google.com/github/GuillermoCarbajal/J-MTPD/blob/main/J-MTPD_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Installation
### Clone Repository
```
git clone https://github.com/GuillermoCarbajal/J-MTPD.git
```


### Download deblurring models

[Trajectory Prediction Network](https://iie.fing.edu.uy/~carbajal/J-MTPD/camera_shake_epoch25_epoch35_epoch50_epoch10_epoch5_epoch25_epoch25_epoch25_epoch27_epoch24_epoch4_epoch10_epoch22_epoch23_epoch90.pkl)           
[Restoration Network](https://iie.fing.edu.uy/~carbajal/J-MTPD/camera_shake_epoch25_epoch35_epoch50_epoch10_epoch5_epoch25_epoch25_epoch25_epoch27_epoch24_epoch4_epoch10_epoch22_epoch23_epoch90_G.pkl)

### Deblur an image or a list of images
```
python test_J-MTPD.py -b blurry_img_path --reblur_model reblur_model_path --restoration_network restoration_model_path --output_folder results
```

### Parameters
Additional options:   
  `--blurry_image`: may be a singe image path or a .txt with a list of images.
  
  `--resize_factor`: input image resize factor (default 1)     
  
  `--focal_length`: given focal length. By default is assummed f=max(H,W). For Kohler dataset images `f=3900`.

## Aknowledgments 
We thank the authors of [Deep Model-Based Super-Resolution with Non-Uniform Blur](https://arxiv.org/abs/2204.10109) for the Blind Deconvolution Network provided in https://github.com/claroche-r/DMBSR 


Guillermo Carbajal was supported partially by Agencia Nacional de Investigacion e Innovación (ANII, Uruguay) `grant POS FCE 2018 1 1007783`. The experiments presented in this paper were carried out using ClusterUY (site: https://cluster.uy).
