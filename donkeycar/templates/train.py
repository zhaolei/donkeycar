#!/usr/bin/env python3
"""
Scripts to train a model using tensorflow or pytorch.
Examples:
  TensorFlow: train.py --tubs data/ --model models/mypilot.h5 --type linear
  PyTorch:    train.py --tubs data/ --model models/mypilot.ckpt --type resnet34 --framework pytorch

Usage:
    train.py [--tubs=tubs] (--model=<model>) [--type=<model_type>]
    [--framework=(tensorflow|pytorch)] [--checkpoint=<checkpoint>]
    [--transfer=<transfer>] [--comment=<comment>]

Options:
    -h --help              Show this screen.
"""

from docopt import docopt
import donkeycar as dk


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    framework = args['--framework'] or getattr(cfg, 'DEFAULT_AI_FRAMEWORK',
                                               'tensorflow')
    checkpoint = args['--checkpoint']
    transfer = args['--transfer']
    comment = args['--comment']
    if framework == 'tensorflow':
        from donkeycar.pipeline.training import train
        train(cfg, tubs, model, model_type, transfer, comment)
    elif framework == 'pytorch':
        from donkeycar.parts.pytorch.torch_train import train
        train(cfg, tubs, model, model_type, checkpoint_path=checkpoint)
    else:
        raise ValueError(
            f"Unknown framework '{framework}', expected tensorflow or pytorch"
        )


if __name__ == "__main__":
    main()
