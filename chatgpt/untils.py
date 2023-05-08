import pickle

def save_pkl(file,name):
    with open(name+'.pickle', 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(name):
    with open(name + '.pickle', 'rb') as handle:
        file = pickle.load(handle)
    return file
