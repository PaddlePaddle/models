

"""Query a MRQA model server to generate predictions.
Usage:
if __name__ == '__main__':
cd baseline
python3.6 serve.py <params.gz> 8888 &  # Start model server in separate process on port 8888
cd ..
python3.6 predict-server.py data.jsonl pred.json 8888  # Query model on data.jsonl, generate pred.json
"""
import argparse
# from allennlp.common.file_utils import cached_path
import gzip
import json
import errno
import requests
import socket
import time
# from allennlp.common.tqdm import Tqdm

OPTS = None


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("dataset")
    parse.add_argument("output_file")
    parse.add_argument("port", type=int)
    args = parse.parse_args()

    all_predictions = {}
    contexts = []
    f = open(args.dataset)
    # single_file_path_cached = cached_path(args.dataset)
    # if single_file_path_cached.endswith('.gz'):
    #     f = gzip.open(single_file_path_cached, 'rb')
    # else:
    #     f = open(single_file_path_cached)
    for example in f:
        context = json.loads(example)
        if 'header' in context:
            continue
        contexts.append(context)
    f.close()

    s = socket.socket()
    for i in range(600): # Try for 10 minutes
        try:
            s.connect(('127.0.0.1', args.port))
        except socket.error as e:
            if e.errno != errno.ECONNREFUSED:
                # Something other than Connection refused means server is running
                break
        time.sleep(1)
    else:
        raise Exception('Could not connect to server')
    s.close()

    # for context in Tqdm.tqdm(contexts,total = len(contexts)):
    results = {}
    cnt1 = cnt2 = 0
    for context in contexts:
        start = time.time()
        cnt1 += 1
        pred = requests.post('http://127.0.0.1:%d' % args.port, json=context)
        print pred
        all_predictions.update(pred.json())

        results.update(pred.json()['results'])
        end=time.time()
        print(end-start)
        # print('send: {}, recv: {}'.format(cnt1, len(results)))
        print(pred.json()['results'])

    with open(args.output_file,'w') as f:
        json.dump(results, f, indent=1)

    # with open(args.output_file,'w') as f:
    #     json.dump(all_predictions,f)

