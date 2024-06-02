import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="A script to handle command-line options for training and evaluating models.\nFirst time you you run this library please run with 'train_model' option."
    )

    # Add an argument for specifying an option
    parser.add_argument(
        '-o', '--option',
        type=str,
        choices=['train_model', 'evaluate_fp32', 'evaluate_fp16', 'int8_PTQ', 'int8_QAT'],
        default = 'None',
        help='Specify the action to perform: train_model, evaluate_fp32, or evaluate_fp16. Use int8_PTQ for Post Training Quantization and int8_QAT for Quantization Aware Training'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Handle the specified option
    if args.option == 'train_model':
        import train_model
    elif args.option == 'evaluate_fp16':
        import quantize_f16  # Assuming this was a typo and you meant model_f16
    elif args.option == 'evaluate_fp32':
        import model_f32
    elif args.option == 'int8_PTQ':
        import quantize_int8_PTQ
    elif args.option == 'int8_QAT':
        import quantize_int8_QAT

    else:
        print("No option specified. Use --help for more information.")

if __name__ == "__main__":
    main()
