# Training-a-CNN-using-caffe
based on  bvlc_reference_caffenet model

# **Deep Learning with Kaggle Cats and Dogs Dataset on MSI**

## I. Training Kaggle Cats and Dogs Dataset on Caffe (Python)

1. Load modules and environment
```
module load caffe/1.0
source activate caffe1.0_lite
```
*Note: If you're lazy like me, you could also add an alias to your `bash.rc` file to call the above set of commands:*
```
alias init_caffe='echo "Loading modules..."; module load caffe/1.0; module list; echo "Loading conda environment..."; source activate caffe1.0_lite'
```
*Now you just need to call `init_caffe` from the terminal.
Also, if you plan to install new packages, you can use a clone environment:*
`conda create --clone <source env> --name <clone env>`

2. Make a directory for the dataset
```
cd kaggle-catdog
mkdir input
cd input
```
3. Download dataset from the [Kaggle Cats vs. Dogs Competition Page](https://www.kaggle.com/c/dogs-vs-cats/data) and extract; if your data is stored elsewhere (like a shared folder), use symbolic links:

`ln -s <target_dir> <link_name>`

4. Edit the pathnames to the datasets in `create_lmdb` and create LMDB creatdataset:
```
cd code
python create_lmdb.py
```
*Note: This code will:*
- run histogram-equalization on all training images, resize all training images to a 227x227 format.
- divide the training data into 2 sets: One for training (5/6 of images) and the other for validation (1/6 of images)
- store the training and validation in 2 LMDB databases (train and val)*

5. Compute image-mean (used to make training data zero-mean)

`compute_image_mean.bin -backend=lmdb ../input/train_lmdb ../input/mean.binaryproto`

6. Pick a model definition (we used the BVLC Caffenet model) and edit the following on the train prototxt model files:

- pathname for input data and mean image
- change number of outputs from 1000 to 2 (as the original model was trained to classify 1000 classes)

7. Update the solver definitions with the new pathnames for `net` and `snapshot`

*Note: Change solver parameters as needed; by default the solver computes the accuracy of the model using the validation set every 1000 iterations; the optimization process will run for a maximum of 20000 iterations, and will take a snapshot of the trained model every 5000 iterations.*

8. Time to train! Since MSI's Caffe is compiled for the GPU, running the training code will result in a CUDA error. You need to create a PBS script that will execute your code on Mesabi's k40 GPU cluster. To do this, edit the included PBS script (train_catdog.pbs):
```
#!/bin/bash -l                                                                                                                                 
#PBS -l nodes=1:ppn=24:gpus=2,walltime=8:00:00                                                                                                
#PBS -q k40                                                                                                                                    
#PBS -m abe                                                                                                                                    
#PBS -M <email_address>                                                                                                       

module load caffe/1.0
source activate <caffe env>

cd <full_path_to_caffe_model_directory>

caffe.bin train --solver solver.prototxt 2>&1 | tee caffe_train.log                                                       
```
*Notes: We use "tee" to redirect output to a log file (as shown above).

If for some reason, training quits (maybe you exceeded walltime limit or something else failed), you can use the snapshots (saved as .solverstate files) to resume training; just replace the previous train command in the PBS script with the following:*
```
caffe.bin train --solver caffe_models/solver.prototxt --snapshot <solverstate_file>
```
9. To submit the job to the queue, use:
```
qsub <PBS_script>
```
