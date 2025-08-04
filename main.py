# main.py

import sys
from training.train_model import run_training

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <model_name>")
        print("Available models: mobilenetv3, nasnetmobile, resnetrs, xception, resnet152, densenet201")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    run_training(model_name)
