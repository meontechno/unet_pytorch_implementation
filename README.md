# unet_pytorch_implementation
Unet image multiclass segmentation 

## Folder structure
train.py (training implementation) \n
predict_video.py \n
predict_image.py \n
/model
 - custom.py (UNet model implementation) \n 
/utils
 - dataset.py (Dataset loader)
 - eval.py (Model evaluation implementation)

## Segmentation 
Total classes 3
    - Background class (floor, windows, blinds, door, ceiling etc) -> mask color encoding black (0, 0, 0)
    - Wall -> mask color encoding red (255, 0, 0)
    - Patch -> mask color encoding green (0, 255, 0)


## Neural Net Architecture: UNet
In this project, UNet is implemented from scratch in model/custom.py
UNet is an end to end fully convolutional network (FCN) used for semantic segmentation.
UNet mainly has two paths:
- Encoder (Feature down sampling): used to capture the context in the image. Encoder
  contains traditional stack of convolutional and max pooling layers.
- Decoder (Feature up sampling): used for precise localization. Decoder contains
  transposed convolutions and regular convolutions


## Dataset preparation
Images and masks are then split into Train, Val and Test sets with ratios 70%, 20%, and 10% respectively.
Additionally image augmentations can used to increase the size of the training set by making slight alterations to the training images (Horizontal flips, Noise, blur, etc).

### Color encoding
background class pixels are masked with color black (0, 0, 0)
Wall class pixels are masked with color red (255, 0, 0)
Patch class pixels are masked with color green (0, 255, 0)



## Training Inputs & Outputs
Input tensor shape: (batch x channels x height x width)
Input tensor example: (16 x 3 x 224 x 224)
where, batch = training batch size
       channels = 3 RGB
       height = Image height 224
       width = Image width 224
Input tensor contains normalized pixel values

Output tensor shape: (batch x num_classes x height x width)
Output tensor example: (16 x 3 x 224 x 224)
where, batch = training batch size
       num_classes = 3
       height = 224 same as input height
       width = 224 same as input width
output tensor contains pixel level probabilities in separate channels for each class.


## Inference Inputs & Outputs
Input tensor shape: (batch x channels x height x width)
Input tensor example: (1 x 3 x 224 x 224)
where, batch = 1
       channels = 3 -> RGB
       height = Image height 224
       width = Image width 224
Input tensor contains normalized pixel values

Output tensor shape: (batch x num_classes x height x width)
Output tensor example: (1 x 3 x 224 x 224)
where, batch = 1
       num_classes = 3
       height = 224 same as input height
       width = 224 same as input width
output tensor contains pixel level probabilities in separate channels for each class.
Additional processing is done to extract list of all predicted classes from the mask.

