# Fully convolutional networks in Keras
In this repository we provide the implementation of fully convolutional networks in Keras for the VGG16, VGG19, InceptionV3, Xception and MobileNetV2 models, for use in various image/keyframe annotation or classification tasks.

## What is a fully convolutional network?
A convolutional network that has no Fully Connected (FC) layers is called a fully convolutional network (FCN). An FC layer has nodes connected to all activations in the previous layer, hence, requires a fixed size of input data. The only difference between an FC layer and a convolutional layer is that the neurons in the convolutional layer are connected only to a local region in the input. However, the neurons in both layers still compute dot products. Since their functional form is identical every FC layer can be replaced by a convolutional layer [2].  

The conversion of FC layers to convolutional ones allows us to *slide* the convolutional network efficiently across many spatial positions in a larger image, in a single forward pass, i.e. it lifts the requirement that the input image is of fixed size (typically 224x244). Additionally, this conversion can in practice be realized by  *reshaping* the weight matrix in each FC layer into the weights of the convolutional layer filters. Therefore, we can directly copy the weights of a model pre-trained on ImageNet. This in turn, allows for faster training times and does not require a large collection of training images (since the FCN does not need to be trained from scratch).  

Because of this *sliding*  of the convolutional network in the image, the FCN produces many decisions, one for each spatial region analysed. To come up with a single decision we add on top of the FCN a global pooling operation layer for spatial data. This can be either a global max pooling layer or a global average pooling layer. The provided FCN models here, use a global max pooling layer; however, the conversion needed to change this to a global average pooling layer is straight-forward.  

Along with the implementation of the FCNs, we also implemented a few variations for feeding square images to an FCN, primarly for comparison of the FCN with the traditional convolutional network architectures that require the input image to be square. These variations preserve the original aspect aspect ratio of the image, by means of cropping or padding. These are:
* `cropping`: crops the largest center part of the image.
* `padding`: pads the smaller dimension of the image with black pixels to convert it to a square image.
* `multi-crop`: crops three overlapping parts that cover the whole area of the image (for more information see [1]).  

In our work [1], we observed that just by converting the VGG16 model to a fully convolutional network and training it on the  two-class [AVA dataset](https://github.com/mtobeiyf/ava_downloader), we achieved an increase in accuracy in the specific problem of assessing the aesthetic quality of images. In the same work, experiments on the aforementioned variations of feeding the images to the FCN (`cropping`, `padding`, `multi-crop`) and experiments utilizing models with skip connections are conducted.

## Prerequisites
* Python (both 2.x and 3.x version are compatible)
* Keras (version 2.2.4 and above)
* Jupyter Notebook

To install Python see [here](https://www.python.org/). Once you have successfully installed Python, you can use the `pip install keras jupyter notebook` command to install all prerequisites.

## Installing the scripts
In this repository we provide the following files:
* "README.md"
* "extensions" directory
* "FCN_setup.py"
* "FCN_demo.ipynb"
* "test_image.jpg"

The FCN implementations of VGG16, VGG19, InceptionV3 and Xception models as well as the variations of feeding the images to the FCN (`cropping`, `padding`, `multi-crop`) are implemented in python scripts and are provided in the "extensions" directory. These files must be installed in the Keras folder in the appropriate locations. To easily install the provided extensions to their respective locations we have included the "setup.py" python script. This will install everything that is needed from the "extensions" directory. Just clone the repository and run `python FCN_setup.py install`. To uninstall the FCN extensions from Keras, run `python FCN_setup.py uninstall`.

*Note that you will have to provide administration privileges in Windows platforms or run the "FCN_setup.py" as a super-user in Linux platforms, for the installation to complete successfully. *

## Demo of FCNs
In the "FCN_demo.ipynb" jupyter notebook a complete framework for constructing and training an FCN model for your custom dataset is implemented. All you need to change are the parameters in the third code cell (titled "Setup parameters") where you can set the training and validation image directories, the number of classes of your dataset, and other hyper-parameters. The notebook will setup everything necessary and will proceed to perform the following experiments:
1. Finetune the original model,
2. Create an FCN version and finetune using the original input size (`original_size`),
3. Finetune the FCN model using 1.5x`original_size`,
4. Finetune the FCN model using 2.0x`original_size`,
5. Finetune the FCN model using 3.0x`original_size`,
6. Finetune the FCN model using 1.5x`original_size` (using the `cropping` method),
7. Finetune the FCN model using 1.5x`original_size` (using the `padding` method),
8. Finetune the FCN model using 1.5x`original_size` (using the `multi-crop` method),
9. Finetune the FCN model using 1.5x`original_size` (using the `multi-crop` method), also introducing a skip connection.

In [1] we observed an increase in accuracy when running experiment #2 compared to the results of experiment #1. Experiment #9 achieved overall the best accuracy compared to the rest of the tests. For more results on the specific clasification problem of assessing the aesthetic quality of photos, see [1]. We would like to stress again that these methods may be applicable to any image annotation or classification problem where avoiding to resize and alter the aspect ratio of the input training/testing image may be beneficial (e.g. image forensic analysis, quality assessment and others).

#### References
[1] *K. Apostolidis, V. Mezaris, “Image Aesthetics Assessment using Fully Convolutional Neural Networks”, Proc. 25th Int. Conf. on Multimedia Modeling (MMM2019), Thessaloniki, Greece, Jan. 2019.*

[2] *J. Long, E. Shelhamer, T. Darrell, “Fully convolutional networks for semantic segmentation”, Proc. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 3431-3440, IEEE, 2015.*

## License and Citation
This code is provided for academic, non-commercial use only. If you find this code useful in your work, please cite the following publication where this implementation of fully convolutional networks is utilized:
*K. Apostolidis, V. Mezaris, “Image Aesthetics Assessment using Fully Convolutional Neural Networks”, Proc. 25th Int. Conf. on Multimedia Modeling (MMM2019), Thessaloniki, Greece, Jan. 2019.*

## Acknowledgements
This work was supported by the European Union Horizon 2020 research and innovation programme under contracts H2020-687786 InVID and H2020-732665 EMMA.
