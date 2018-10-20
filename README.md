# DeblurGAN
[arXiv Paper Version](https://arxiv.org/pdf/1711.07064.pdf)

Pytorch implementation of the paper DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.

Our network takes blurry image as an input and procude the corresponding sharp estimate, as in the example:
<img src="images/animation3.gif" width="400px"/> <img src="images/animation4.gif" width="400px"/>


The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)

## How to run

### Prerequisites
- NVIDIA GPU + CUDA CuDNN (CPU untested, feedback appreciated)
- Pytorch (installation guide with python3.5. We need to install torch-0.3.1 instead of torch-0.4.1, otherwise there will be error)
```bash
    pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
    pip3 install torchvision
```    
- Install requried libraries
```bash
    pip3 install dominate
    pip3 install visdom
```
Download weights from [Dropbox](https://www.dropbox.com/s/5r6cy0x72s8x9yf/latest_net_G.pth?dl=0) . Note that during the inference you need to keep only Generator weights. Or directly copy the link https://www.dropbox.com/s/5r6cy0x72s8x9yf/latest_net_G.pth and "wget" it from Linux.

Put the weights into 
```bash
/.checkpoints/experiment_name
```
To test a model put your blurry images into a folder and run:
```bash
python test.py --dataroot /.path_to_your_data --model test --dataset_mode single --learn_residual
```
## Data
Download dataset for Object Detection benchmark from [Google Drive](https://drive.google.com/file/d/1CPMBmRj-jBDO2ax4CxkBs9iczIFrs8VA/view?usp=sharing)

## Train

If you want to train the model on your data run the following command to create image pairs:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data --width 160 --height 120
```
The folder_A can be your real data, such as

<img src="https://github.com/lindylin1817/DeblurGAN/blob/master/CX211.jpg" width="120px"/>
and folder_B can be your self-generated fake data, such as

<img src="https://github.com/lindylin1817/DeblurGAN/blob/master/CX211_B.jpg" width="120px"/>
and the folder_AB will be the output of the combined images path. The output will look like the following picture.

<img src="https://github.com/lindylin1817/DeblurGAN/blob/master/CX211_AB.jpg" width="120px"/>
"--width" in the args is the resized width of resolution, and "--height" is the resized height of resolution. The modifined "combine_A_and_B.py" will automatically resize the images in folder_A and folder_B to ensure the pair of images will have same resolution.

And then the following command to train the model

```bash
python train.py --name name_of_folder_to_store_trained_model --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --gpu_ids 0,1,2,3 --fineSize CROP_SIZE (we used 256)
```
If you only have one GPU with id=0, then just set "--gpu_ids 0". If you train with CPU, set gpu_ids=-1. The trained models will stored under ./checkpoints/name_of_folder_to_store_trained_model

## Other Implementations

[Keras Blog](https://blog.sicara.com/keras-generative-adversarial-networks-image-deblurring-45e3ab6977b5)

[Keras Repository](https://github.com/RaphaelMeudec/deblur-gan)



## Citation

If you find our code helpful in your research or work please cite our paper.

```
@article{DeblurGAN,
  title = {DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks},
  author = {Kupyn, Orest and Budzan, Volodymyr and Mykhailych, Mykola and Mishkin, Dmytro and Matas, Jiri},
  journal = {ArXiv e-prints},
  eprint = {1711.07064},
  year = 2017
}
```

## Acknowledgments
Code borrows heavily from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The images were taken from GoPRO test dataset - [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur_release)


