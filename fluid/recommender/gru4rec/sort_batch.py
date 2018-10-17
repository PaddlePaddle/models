def batch(reader, batch_size, sort_group_size, drop_last=False):
    """
    Create a batched reader.
    :param reader: the data reader to read from.
    :type reader: callable
    :param batch_size: size of each mini-batch
    :type batch_size: int
    :param sort_group_size: size of partial sorted batch
    :type sort_group_size: int
    :param drop_last: drop the last batch, if the size of last batch is not equal to batch_size.
    :type drop_last: bool
    :return: the batched reader.
    :rtype: callable
    """
    	
    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == sort_group_size:
                sortl = sorted(b, key=lambda x:len(x[0]), reverse=True)
		b = []
		c = []
		for sort_i in sortl:
		    c.append(sort_i)
		    if(len(c) == batch_size):
		        yield c
                        c = []
        if drop_last == False and len(b) != 0:
            sortl = sorted(b, key=lambda x:len(x[0]), reverse=True)
	    c = []
	    for sort_i in sortl:
		c.append(sort_i)
		if(len(c) == batch_size):
		    yield c
                    c = []
    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size should be a positive integeral value, "
                         "but got batch_size={}".format(batch_size))
    return batch_reader
