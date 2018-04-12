from NB import *

if __name__ == '__main__':
    nb = NB('./data.csv', 'user_dict.utf8', 'stop_words.utf8', 15)
    nb.train()
