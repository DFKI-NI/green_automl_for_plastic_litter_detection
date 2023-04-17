# Green-AutoML for Plastic Litter Detection
Code for the paper "Green-AutoML for Plastic Litter Detection" published at ICLR 2023 Workshop: Tackling Climate Change with Machine Learning.

1. Clone the git repository and install the packages defined in the "requirements.txt" file.

2. To train the standard architectures, run:

            ``python standard_architectures/hpo_outer_loop.py --model alexnet --path_to_data *path_to_data*``

            The data is not publicaly available. You will need to provide your own dataset. You can repreat this with all five models of the paper: alexnet,      efficient_net, mobile_net, resnet or vgg. This will automatically call "train.py" several times to tune the hyperparameters. You can find the results including the metrics on the validation data, the training meissions and the model in the ouput_dir.

3. Perform pruning on the best run per model: 

            ``python standard_architectures/pruning.py --model alexnet --path_to_data *path_to_data* --path_to_best_trial *path_to_best_trial*``

            In addition to the arguments of the "train.py", you need to provide the path to the best trial from training. This will return the final metrics on the test set of the model.

4. Run Efficient Neural architecture Search:

            ``python ENAS.py --path_to_data *path_to_data*``

            This will perform Neural Architecture Search and train the best archietcture from scratch. It provides the final metrics on the test set and the training emissions. 
