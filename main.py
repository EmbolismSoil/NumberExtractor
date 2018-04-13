from NB import *

if __name__ == '__main__':
    nb = NB('./black_qq_sms.csv', './user.dict.utf8', './stop_words.utf8', 10)
    #nb.train()
    print(nb.predict('法并公布牛股,不收.费人满即封!加Q裙:验证168必填,退订回T'))