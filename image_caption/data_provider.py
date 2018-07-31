import string
import os

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# extract descriptions for images
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			# desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			# desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word != '.']
			# store as string
			desc_list[i] =  ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append('data/Flicker8k_Dataset/' + key + '.jpg' + '\t' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

filename = 'data/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
# print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
# print('Vocabulary Size: %d' % len(vocabulary))
# save to file
save_descriptions(descriptions, 'data/descriptions.txt')


with open('data/descriptions.txt','r') as f:
	a = f.readlines()
	b = []
	with open('data/final_data.txt','w') as f1:
		i = 1
		for line in a:
			if (i-1) % 5 == 0:
				img_path, lab = line.strip().split('\t')
				b.append(img_path)
				f1.write(img_path+'\t'+lab+'\n')	
			i += 1

with open('data/Flickr_8k.trainImages.txt','r') as f1:
	pic0 = f1.readlines()
	pic = []
	for i in pic0:
		pic.append(i.strip())
	# print(pic[0])

	with open('data/final_data.txt', mode='r') as f:
		lines = f.readlines()
		with open('data/train.txt','w') as f2:

			for line in lines:
				line_split = line.strip().split('\t')
				if line_split[0][23:] in pic:
					print(line_split[0][23:])
					f2.write(line)



with open('data/Flickr_8k.testImages.txt','r') as f1:
	pic0 = f1.readlines()
	pic = []
	for i in pic0:
		pic.append(i.strip())
	# print(pic[0])

	with open('data/final_data.txt', mode='r') as f:
		lines = f.readlines()
		with open('data/test.txt','w') as f2:

			for line in lines:
				line_split = line.strip().split('\t')
				if line_split[0][23:] in pic:
					print(line_split[0][23:])
					f2.write(line)

os.remove('data/descriptions.txt')