
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='This is a demonstration program')

    # Add an argument to the parser, specifying the expected type, help message, etc.
    parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

    return parser.parse_args()

def main():
    args = parse_args()

    # Now we can use the argument value in our program
    print(f"batch size: {args.batch_size}")  # Corrected to args.llm

if __name__ == '__main__':
    main()
