from NB import *

if __name__ == '__main__':
    nb = NB('./black_qq_sms.csv', './user.dict.utf8', './stop_words_cn.utf8', './noise_char.utf8', 5, 2)
    #nb.train()
    nb.test('./black_qq_sample.csv')