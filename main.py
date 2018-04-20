from NB import *

if __name__ == '__main__':
    nb = NB('./black_qq_sample_filtered_20180419.csv', './user.dict.utf8', './stop_words_cn.utf8', './noise_char.utf8', 5, 2)
    nb.train()
    #nb._predict(['出行', '参加', '及时', '团委', '联系', '邓艳艳', '预祝'])
    #nb.test('./black_qq_sample.csv')