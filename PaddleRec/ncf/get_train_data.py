import scipy.sparse as sp
import numpy as np
from time import time
import args

def get_train_data(filename, write_file, num_negatives):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    

        file = open(write_file, 'w') 
        print("writing " + write_file)
        
        for (u, i) in mat:
            # positive instance
            user_input = str(u)
            item_input = str(i)
            label = str(1)
            sample =  "{0},{1},{2}".format(user_input, item_input,label) + "\n"
            file.write(sample)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in mat.keys():
                    j = np.random.randint(num_items)
                user_input = str(u)
                item_input = str(j)
                label = str(0)
                sample =  "{0},{1},{2}".format(user_input, item_input,label) + "\n"
                file.write(sample)
                
if __name__ == "__main__":
    args = args.parse_args()
    get_train_data(args.path + args.dataset + ".train.rating", args.train_data_path, args.num_neg)
    