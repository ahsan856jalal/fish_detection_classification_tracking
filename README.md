# fish_detection_classification_tracking
This repo is for the research work "A system paper on fish detection, classification and fish relative abundance estimation under unconstrained underwater videos"

# Guide for Usage of Fish Detection, Classification & Tracking on videos code:

### Installation of Conda:
Conda is required to set-up the environment for this repository. If it isn’t installed, please do so by following the instructions on [this link](https://docs.anaconda.com/anaconda/install/).

### Setting up the code:
Clone the code repository by running the following command in the desired location:


```
    git clone https://github.com/ahsan856jalal/fish_detection_classification_tracking.git
    cd fish_detection_classification_tracking
```

It is recommended to use a conda environment, which can be created by running the following command:

```
conda env create -f environment.yml
```

For google Colab, alternatively, following command can be run to install all the required dependencies:

```
pip install -r requirements.txt
```

Once the repository has been cloned, place the files provided on google drive [link] (https://drive.google.com/drive/folders/1GxfMIrP9S9rijaL50OnDxDAAhiGPW12n?usp=sharing) in `fish_detection_classification_tracking/trained/`

### Usage:
Before running, if a conda environment is being used, please activate it by running the following command:

```    conda activate fish```

From the root directory of the cloned repository, run the following command for inference.

```    ./detect.sh {Path to video/image or folder containing videos or images} {Output Path}```
For example 
```    ./detect.sh A000002_R.avi.46084 results/```

Once the code runs completely, a file named `detections.csv` will be stored in {Output Path}. For each object found in a video/image the following information is output as a row to the CSV file:

``` <video_name> <frame_no> <row1> <col1> <row2> <col2> <row3> <col3> <row4> <col4> <fish_specie> <tracking_id> ```

Note: The provided path can contain both videos and images at the same time. For images, the frame_no will always be zero in 'detections.csv'

# Guide for Usage of Spatio-temporal Fish Detection:
This part guides you to run the spatio-temporal based fish detection part to get the cited accuracies and F-scores.
### Setting up the code:

place the data files provided on google drive [link](https://drive.google.com/drive/folders/1ro04nd8yyHsOJb66JZz-eCMvvZHexT3V?usp=sharing) in `fish_detection_classification_tracking/data/`

 Run ``` python kmeans_on_hist.py ``` to get kmeans based color segmentation on Optical flow histogram equilized data.
### Build YOLOv4
clone the yolov4 repo  in _fish_detection_classification_tracking_ folder```git clone https://github.com/AlexeyAB/darknet.git yolo_framework ```
```cd yolo_framework```
```vim Makefile```
Edit first 9 lines with this
``` GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=1
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
Then run ``` make ```

After successful build run ``` cd .. ```

### Usage

Run ``` python optical_results.py ``` This will apply fish detector on optical_kmeans blobs and save text files

Now we will run ``` python merging_yolo_optical_hist_text.py ``` to merge Optical_kmeans results with YOLO results

Run ``` python optical_yolo_merge_fscore.py ``` to calculate F-score and Accuracy values on the test set as cited in the paper




