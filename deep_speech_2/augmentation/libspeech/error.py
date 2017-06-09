"""
This submodule contains custom errors and warnings used across libspeech.
"""
from contextlib import contextmanager
from subprocess import CalledProcessError


class ConfigurationError(Exception):
    """
    Exception indicating an issue in the passed-in config file
    """
    pass


class PartitionError(Exception):
    """
    Indicates an issue with requested slurm partition.
    """
    pass


class SampleRateError(Exception):
    """
    When a sample rate-related error is encountered (e.g., attempting
    to add two segments with different sample rates.)
    """
    pass


class AudioLengthError(Exception):
    """
    When a signal length-related error is encountered (e.g., attempting
    to add two segments with different lengths.)
    """
    pass


class AudioFormatError(Exception):
    """
    When an audio format-related error is encountered (e.g. Trying to
    open a pcm file as a wav file.)
    """
    pass


class SampleRateWarning(UserWarning):
    """
    Warning to notify user that some non-breaking sample rate issue is
    present.
    """


class ClippingWarning(UserWarning):
    """
    Warning to notify user of clipped samples.
    """
    pass


class NormalizationWarning(UserWarning):
    """
    Warning to notify user of audio normalization issues.
    """
    pass


class ConfigurationWarning(UserWarning):
    """
    Warning to notify user of configuration issues.
    """
    pass


class CalledProcessErrorWithOutput(Exception):
    """
    This is identical to subprocess.CalledProcessError except that it
    prints the subprocess output as well.  There's also an optional `msg`
    argument to provide additional information.

    Usage:
        >>> try:
        >>>     output = subprocess.check_output(cmd)
        >>> except subprocess.CalledProcessError as exc:
        >>>     raise CalledProcessErrorWithOutput(exc)
    """

    def __init__(self, called_process_error, msg=None):
        """
        Args:
            called_process_error (subprocess.CalledProcessError)
            msg (str): optional message
        """
        self.cmd = called_process_error.cmd
        self.returncode = called_process_error.returncode
        self.output = called_process_error.output
        self.msg = msg

    def __str__(self):
        if self.msg:
            return ("{}, Command '{}' returned non-zero exit status {}, "
                    "output '{}'"
                    .format(self.msg, self.cmd, self.returncode, self.output))
        else:
            return ("Command '{}' returned non-zero exit status {}, "
                    "output '{}'"
                    .format(self.cmd, self.returncode, self.output))


class CharMapWarning(UserWarning):
    """
    Warning to notify users of presence of chars not in the model's
    char map
    """
    pass


@contextmanager
def raise_CalledProcessErrorWithOutput():
    """
    Raises a CalledProcessErrorWithOutput in place of a CalledProcessError.
    The CalledProcessErrorWithOutput prints subprocess output as well as
    the returncode and command information.

    Usage:
    >>> # Normal CalledProcessError:
    >>> raise CalledProcessError(returncode=1, cmd='ls', output='Error!')
    CalledProcessError: Command 'ls' returned non-zero exit status 1

    >>> # Using this context manager:
    >>> with raise_CalledProcessErrorWithOutput():
    >>>     raise CalledProcessError(returncode=1, cmd='ls', output='Error!')
    CalledProcessErrorWithOutput
    "Command 'ls' returned non-zero exit status 1", output 'Error!'
    """
    try:
        yield
    except CalledProcessError as exc:
        raise CalledProcessErrorWithOutput(exc)
