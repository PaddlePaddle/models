import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter, REMAINDER

import yaml


class ColorTTY(object):
    def __init__(self):
        super(ColorTTY, self).__init__()
        self.colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']

    def __getattr__(self, attr):
        if attr in self.colors:
            color = self.colors.index(attr) + 31

            def color_message(message):
                return "[{}m{}[0m".format(color, message)

            setattr(self, attr, color_message)
            return color_message

    def bold(self, message):
        return self.with_code('01', message)

    def with_code(self, code, message):
        return "[{}m{}[0m".format(code, message)


def parse_args(argv):
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument("-o", "--opt", nargs=REMAINDER, help="set configuration options")
    args = parser.parse_args(argv)

    if args.config is None:
        print("Please specify config file")
        sys.exit(1)

    cli_config = {}
    if 'opt' in vars(args) and args.opt is not None:
        for s in args.opt:
            s = s.strip()
            k, v = s.split('=')
            if '.' not in k:
                cli_config[k] = v
            else:
                keys = k.split('.')
                cli_config[keys[0]] = {}
                cur = cli_config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]

    args.cli_config = cli_config
    return args
