# Adaround-Tools
AdaRound tools for per layer quantization


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

Use for more information:
`
python main.py --help
`



