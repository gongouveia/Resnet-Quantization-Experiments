# Resnet_quantization-Tools
tools for per layer quantization


# Model Management Script

## Overview

This script is developed to manage command-line options for training and evaluating machine learning models. By specifying various options, users can train a new model, evaluate a model with FP32 precision, evaluate a model with FP16 precision, perform 8bit Post Training Quantization (PTQ), or conduct Quantization Aware Training (QAT). Python's argparse library is employed to streamline the handling of command-line arguments effectively.

## Usage

To run the script, use the following command format:

`
python main.py --option <option_name>
`

The <option_name> should be one of the following:



`train_model` to train a new model.

`evaluate_fp32` to evaluate a model in FP32 precision. Requires to run 
`train_model` first.

`evaluate_fp16` to evaluate a model in FP16 precision. Requires to run 
`train_model` first.

`int8_PTQ` to perform Post Training Quantization. Requires to run 
`train_model` first.

`int8_QAT` to perform Quantization Aware Training.


Use for more information:
`
python main.py --help
`
## Models information


## How to contrinute

Contributions to this project are welcome! If you'd like to contribute to add new quantization methods, please follow the standard GitHub workflow:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.
