# Guide for Usage of Fish Detection & Classification code:

### Installation of Conda:
Conda is required to set-up the environment for this repository. If it isn’t installed, please do so by following the instructions on [this link](https://docs.anaconda.com/anaconda/install/).

### Setting up the code:
Clone the code repository by running the following command in the desired location:


```
    git clone https://bitbucket.org/fish-analyse/detection.git
    cd detection
```

It is recommended to use a conda environment, which can be created by running the following command:

```
conda env create -f environment.yml
```

For google Colab, alternatively, following command can be run to install all the required dependencies:

```
pip install -r requirements.txt
```

Once the repository has been cloned, place the files provided on google drive [link](https://drive.google.com/drive/folders/1Likirnxy4cFivSAD_mcs1WdXUyNzUese?usp=sharing) in `detection/trained/`

### Usage:
Before running, if a conda environment is being used, please activate it by running the following command:

```    conda activate fish```

From the root directory of the cloned repository, run the following command for inference.

```    ./detect.sh {Path to video/image or folder containing videos or images} {Output Path}```

Once the code runs completely, a file named `detections.csv` will be stored in {Output Path}. For each object found in a video/image the following information is output as a row to the CSV file:

``` <video_name> <frame_no> <row1> <col1> <row2> <col2> <row3> <col3> <row4> <col4> <fish_specie> <tracking_id> ```

Note: The provided path can contain both videos and images at the same time. For images, the frame_no will always be zero in 'detections.csv'
