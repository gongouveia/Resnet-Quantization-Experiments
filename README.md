# Resnet_quantization-Tools
tools for per layer quantization


# Model Management Script

## Overview

This script is designed to handle command-line options for training and evaluating machine learning models. By specifying different options, users can train a new model, evaluate a model in FP32 precision, or evaluate a model in FP16 precision. The script leverages Python's `argparse` library to manage command-line arguments efficiently.

## Usage

To run the script, use the following command format:

`
python main.py --option <option_name>
`

The <option_name> should be one of the following:



`train_model` to train a new model.

`evaluate_fp32` to evaluate a model in FP32 precision.

`evaluate_fp16` to evaluate a model in FP16 precision.

`int8_PTQ` to perform Post Training Quantization.

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
