from NB import *

if __name__ == '__main__':
    nb = NB('./black_qq_sms.csv', './user.dict.utf8', './stop_words_cn.utf8', './noise_char.utf8', 5, 2)
    nb.train()
    #nb._predict(['天', '请', '收到', '下发', '订购'])
    print(nb.test('./sms_sample_20180413.csv'))