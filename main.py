from NB import *

if __name__ == '__main__':
    nb = NB('./black_qq_sms.csv', './user.dict.utf8', './stop_words.utf8', 10)
    #nb.train()
    print(nb.predict(['请', '加', '我', 'qq', '详谈']))
