def read_txt(videoTxt):
    with open(videoTxt, 'r') as f:
        videolist = f.readlines()
    return videolist


def read_txt_to_index(file):
    data = read_txt(file)
    data = list(map(int, data))
    return data


def main():
    file = 'data_dir/FlyingChairs_release/FlyingChairs_train_val.txt'
    data = read_txt_to_index(file)
    data = list(map(int, data))
    print(data)
    print(len(data))


if __name__ == '__main__':
    main()
