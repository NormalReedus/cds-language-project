# Classifying Hate Speech with BERT

## Prerequisites
You will need to have Bash and Python 3 installed on your computer. These scripts have been tested with Python 3.9.1 on Windows 10 and Python 3.8.6 on Linux (Ubuntu flavour).

## Installation
First, clone the repository somewhere on your computer. Then open a Bash terminal in the root directory of the cloned repository, and run the correct Bash script to generate your virtual environment, required directories, and install dependencies as follows:

**Windows:**
```bash
./create_venv_win.sh
```

**Mac / Linux:**
```bash
./create_venv_unix.sh
```

If you have any issues with this last step, make sure your terminal is running as administrator (on Windows) and that you can execute the Bash scripts with:

**Windows:**
```bash
chmod +x create_venv_win.sh
```

**Mac / Linux:**
```bash
chmod +x create_venv_unix.sh
```

## Running the script
First, make sure the newly created virtual environment is activated with:

**Windows:**
```bash
source lang101/Scripts/activate
```

**Mac / Linux:**
```bash
source lang101/bin/activate
```

Your repository should now have at least these files and directories:

```
data/
lang101/
output/
utils/
1_train_model.py
create_venv_unix.sh
create_venv_win.sh
get-pip.py
README.md
requirements.txt
```

**NOTE:** Going forwards we will assume you have an alias set for running python such that you will not have to type python3 if that is normally required on your system (usually Mac / Linux). If this is not the case, you will have to substitute python with python3 in the following commands.

1.  `./data/` should already have the annotated hate speech dataset when cloning this repository
    - This dataset is not available anywhere else, as it is hand-crafted
2. You can run `python 1_train_model.py` to train the model
    - You can change the batch size with the flag `-b` or `--batch_size`. Default batch size is 32. This heavily influences RAM usage during training. We recommend setting this to < 5 if running the script on a system with 16GB RAM
    - You can change the number of epochs to train with the flag `-e` or `--epochs`. Default is 3
    - Since this process can take a while you can run the script on a sample set of files (for demo purposes if the dataset is too large) with the flag `-s` or `--sample_num`. The default is `None`, which will run the training on the whole dataset
    - For example:
    ```bash
    python 1_train_model.py -e 5 -s 200 -b 5
    ```
3. Your terminal should show a readout of training progress and finally the validation loss and accuracy
4. These measurements are also saved as a history graph, which you can find in `./output/`. This repo already comes with an example history graph from 3 epochs of training on the full dataset. Note that the graph will only be able to show anything when training > 1 epoch, since it needs several points to draw lines between
5. You can also find the saved model in `./output/`