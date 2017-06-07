""" Impulse response block usage

    {
        "type": "impulse_response",
        "rate": // Usage rate between 0.0 and 1.0.
        "ir_dir": // root impulse response database directory.
        "index_file": // impulse reponse index file.
        "tags": // Not specifying this field selects all audio samples in
                // the index.
        "tag_distr": // A dictionary describing the noise type distribution;
                     // maps from a tag in "tags" to a probability mass.
                     // The masses are automatically normalized.
    }
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base
from . import audio_database
from libspeech.audio import SpeechDLSegment


class ImpulseResponseModel(base.ModelInterface):
    """ Instantiates an impulse response model

    Attributes:
        ir_dir (func[int->str]): directory containing impulse responses
        tags (func[int->list]): optional parameter for specifying what
            particular impulse responses to apply.
        tag_distr (func[int->dict]): optional noise distribution
    """

    def __init__(self, ir_dir, index_file, tags=None, tag_distr=None):
        # Define all required parameter maps here.
        self.ir_dir = ir_dir
        self.index_file = index_file

        # Optional parameters.
        if tags is None:
            tags = base.parse_parameter_from(None)
        if tag_distr is None:
            tag_distr = base.parse_parameter_from(None)
        self.tags = tags
        self.tag_distr = tag_distr

        self.audio_index = audio_database.AudioIndex()

    def _init_data(self, iteration):
        """ Preloads stuff from disk in an attempt (e.g. list of files, etc)
        to make later loading faster. If the data configuration remains the
        same, this function does nothing.

        Args:
            iteration (int): current iteration
        """
        self.audio_index.refresh_records_from_index_file(
            self.ir_dir(iteration),
            self.index_file(iteration), self.tags(iteration))

    def transform_audio(self, audio, text, iteration, rng):
        """ Convolves the input audio with an impulse response.

        Args:
            audio (SpeechDLSegment): input audio
            text (str): audio transcription
            iteration (int): current iteration
            rng (random.Random): RNG to use for augmentation.

        Returns:
            (SpeechDLSegment, str, int)
                - convolved audio where the output is at the same average power
                  as the input
                - label
                - number of bytes read from disk
        """
        # This handles the cases where the data source or directories change.
        self._init_data(iteration)

        read_size = 0
        tag_distr = self.tag_distr(iteration)
        if not self.audio_index.has_audio(tag_distr):
            if tag_distr is None:
                if not self.tags(iteration):
                    raise RuntimeError("The ir index does not have audio "
                                       "files to sample from.")
                else:
                    raise RuntimeError("The ir index does not have audio "
                                       "files of the given tags to sample "
                                       "from.")
            else:
                raise RuntimeError("The ir index does not have audio "
                                   "files to match the target ir "
                                   "distribution.")
        else:
            # Querying with a negative duration triggers the index to search
            # from all impulse responses.
            success, record = self.audio_index.sample_audio(
                -1.0, rng=rng, distr=tag_distr)
            if success is True:
                _, read_size, ir_fname = record
                ir_wav = SpeechDLSegment.from_wav_file(ir_fname)
                audio = audio.convolve(ir_wav, allow_resampling=True)
        return audio, text, read_size
