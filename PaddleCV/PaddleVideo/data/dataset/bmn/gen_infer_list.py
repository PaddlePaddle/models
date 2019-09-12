'''
Generate samples for inference from validation dataset.
'''

import json
annos = json.load(open("activitynet_1.3_annotations.json"))
infer_data = {}
count = 0
for video_name in annos.keys():
    if annos[video_name]["subset"] == 'validation':
        infer_data[video_name] = annos[video_name]
        count += 1
        if count == 5:
            break
with open('infer.list.json', 'w') as f:
    json.dump(infer_data, f)
