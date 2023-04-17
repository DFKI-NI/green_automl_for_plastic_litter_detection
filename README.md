# Green-AutoML for Plastic Litter Detection
Code for the paper "Green-AutoML for Plastic Litter Detection" published at ICLR 2023 Workshop: Tackling Climate Change with Machine Learning.

## Abstract
The world’s oceans are polluted with plastic waste and the detection of it is an important step toward removing it. Wolf et al. (2020) created a plastic waste dataset
to develop a plastic detection system. Our work aims to improve the machine learning model by using Green Automated Machine Learning (AutoML). One aspect of Green-AutoML is to search for a machine learning pipeline, while also minimizing the carbon footprint. In this work, we train five standard neural architectures for image classification on the aforementioned plastic waste dataset. Subsequently, their performance and carbon footprints are compared to an Efficient Neural Architecture Search as a well-known AutoML approach. We show the potential of Green-AutoML by outperforming the original plastic detection system by 1.1% in accuracy and using 33 times fewer floating point operations at inference, and only 29% of the carbon emissions of the best-known baseline. This shows the large potential of AutoML on climate-change relevant applications and at the same time contributes to more efficient modern Deep Learning systems, saving substantial resources and reducing the carbon footprint.

## Running the code
1. Clone the git repository and install the packages defined in the "requirements.txt" file.

2. To train the standard architectures, run:
 
            python standard_architectures/hpo_outer_loop.py --model alexnet --path_to_data *path_to_data*

      The data is not publicaly available. You will need to provide your own dataset. You can repeat this with all five models of the paper: alexnet, efficient_net, mobile_net, resnet or vgg. This will automatically call "train.py" several times to tune the hyperparameters. You can find the results including the metrics on the validation data, the training emissions and the model in the ouput_dir.

3. Perform pruning on the best run per model: 

            python standard_architectures/pruning.py --model alexnet --path_to_data *path_to_data* --path_to_best_trial *path_to_best_trial*

      In addition to the arguments of the "train.py", you need to provide the path to the best trial from training. This will return the final metrics on the test set of the model.

4. Run Efficient Neural architecture Search:

            python ENAS.py --path_to_data *path_to_data*

      This will perform Neural Architecture Search and train the best archietcture from scratch. It provides the final metrics on the test set and the training emissions. 

## License
Project is released under the BSD-3 clause.

## Maintenance
The maintainer of this software is Daphne Theodorakopoulos (daphne.theodorakopoulos@dfki.de).

## Cite
