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
        choices=['train_model', 'evaluate_fp32', 'evaluate_fp16'],
        required=True,
        help='Specify the action to perform: train_model, evaluate_fp32, or evaluate_fp16'
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
    else:
        print("Invalid option specified. Use --help for more information.")

if __name__ == "__main__":
    main()
