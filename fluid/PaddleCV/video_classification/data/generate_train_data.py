import os
import cPickle

# read class file
dd = {}
f = open('ucfTrainTestlist/classInd.txt')
for line in f.readlines():
    label, name = line.split()
    dd[name.lower()] = int(label) - 1
f.close()


def generate_pkl(mode):
    # generate pkl
    path = '%s/' % mode
    savepath = '%s_pkl/' % mode
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    fw = open('%s.list' % mode, 'w')
    for folder in os.listdir(path):
        vidid = folder.split('_', 1)[1]
        this_label = dd[folder.split('_')[1].lower()]
        this_feat = []
        for img in sorted(os.listdir(path + folder)):
            fout = open(path + folder + '/' + img, 'rb')
            this_feat.append(fout.read())
            fout.close()

        res = [vidid, this_label, this_feat]

        outp = open(savepath + vidid + '.pkl', 'wb')
        cPickle.dump(res, outp, protocol=cPickle.HIGHEST_PROTOCOL)
        outp.close()

        fw.write('data/%s/%s.pkl\n' % (savepath, vidid))
    fw.close()


generate_pkl('train')
generate_pkl('test')
