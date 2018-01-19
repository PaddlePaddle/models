from utils import RandomScheduleGenerator


def gen_schedule_data(reader,
                      schedule_type="linear",
                      decay_a=0.75,
                      decay_b=1000000):
    """
    Creates a data reader for scheduled sampling.

    Output from the iterator that created by original reader will be
    appended with "true_token_flag" to indicate whether to use true token.

    :param reader: the original reader.
    :type reader: callable
    :param schedule_type: the type of sampling rate decay.
    :type schedule_type: str
    :param decay_a: the decay parameter a.
    :type decay_a: float
    :param decay_b: the decay parameter b.
    :type decay_b: float

    :return: the new reader with the field "true_token_flag".
    :rtype: callable
    """
    schedule_generator = RandomScheduleGenerator(schedule_type, decay_a,
                                                 decay_b)

    def data_reader():
        for src_ids, trg_ids, trg_ids_next in reader():
            yield src_ids, trg_ids, trg_ids_next, \
                  [0] + schedule_generator.processBatch(len(trg_ids) - 1)

    return data_reader


feeding = {
    'source_language_word': 0,
    'target_language_word': 1,
    'target_language_next_word': 2,
    'true_token_flag': 3
}
