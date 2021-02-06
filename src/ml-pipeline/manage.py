import argparse



def run(args):
    """
    TODO:
        watchdog on settings.PIPELINE_SOURCE for changes
        process file
        store results (in directory?)
        move processed file to processed directory
    """
    raise NotImplementedError('Run not implemented')

def train(args):
    raise NotImplementedError('Train not implemented')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()
    sp_run = sp.add_parser('run', help='Runs the ML Pipeline')
    sp_run.set_defaults(func=run)

    sp_train = sp.add_parser('train', help='Trains the ML Pipeline')
    sp_train.set_defaults(func=train)
    args = parser.parse_args()
    command = args.func
    command(args)