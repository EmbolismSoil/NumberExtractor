from NB import *

if __name__ == '__main__':
    nb = NB('./black_qq_sms.csv', './user.dict.utf8', './stop_words.utf8', 5, 2)
    nb.train()
    print(nb._predict(['薪酬', '现结', '咨询', '解扣', 'q']))
    print(nb.test('./black_qq_sms_test.csv'))