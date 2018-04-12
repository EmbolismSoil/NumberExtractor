import re
import jieba
from .models import *
import math


class NB(object):
    def __init__(self, data_path, dict_path, stop_words_path, ctx_len):
        self._data_path = data_path
        self._stop_wrods_path = stop_words_path
        self._ctx_len = ctx_len
        self._data_file = open(data_path, 'r')
        self._stop_wrods_file = open(stop_words_path, 'r')
        jieba.set_dictionary(dict_path)
        self._stop_wrods = set()
        for line in self._stop_wrods_file:
            self._stop_wrods.add(line)

        self._cls_cnt = {False: 0, True: 0}
        self._cls_word_cnt = {False: {}, True: {}}
        self._session = DBSession()


    def update_train(self):
        self.__do_train()
        self.__update()


    def train(self):
        self.__do_train()
        self.__save()


    def __do_train(self):
        cls_cnt = {False: 0, True: 0}
        cls_word_cnt = {False: {}, True: {}}

        for ctx, is_qq in self.__records():
            cls_cnt[is_qq] = cls_cnt.get(is_qq, 0) + 1
            for w in ctx:
                cls_word_cnt[is_qq][w] = cls_word_cnt[is_qq].get(w, 0) + 1

        self._cls_cnt = cls_cnt
        self._cls_word_cnt = cls_word_cnt


    def __save(self):

        is_qq_cls_cnt = ClassCount(cls='is_qq_number', cnt=self._cls_cnt[True])
        is_not_qq_cls_cnt = ClassCount(cls='is_not_qq_number', cnt=self._cls_cnt[False])
        self._session.add(is_qq_cls_cnt)
        self._session.add(is_not_qq_cls_cnt)

        is_qq_word_statis = self._cls_word_cnt[True]
        is_not_qq_word_statis = self._cls_word_cnt[False]

        for k, v in is_qq_word_statis.items():
            cls_word_cnt = ClassWordCount(cls='is_qq_number', cnt=v)
            self._session.add(cls_word_cnt)


        for k, v in is_not_qq_word_statis.items():
            cls_word_cnt = ClassWordCount(cls='is_not_qq_number', cnt=v)
            self._session.add(cls_word_cnt)

        #let it crash
        self._session.commit()


    def __update(self):
        is_qq_number_cnt = self._session.query(ClassCount).filter_by(cls='is_qq_number').one()
        is_qq_number_cnt = is_qq_number_cnt.cnt
        is_not_qq_number_cnt = self._session.query(ClassCount).filter_by(cls='is_not_qq_number').one()
        is_not_qq_number_cnt = is_not_qq_number_cnt.cnt

        self._cls_cnt[True] = is_qq_number_cnt + self._cls_cnt[True]
        self._cls_cnt[False] = is_not_qq_number_cnt + self._cls_cnt[False]

        is_qq_word_statis = self._cls_word_cnt[True]
        is_not_qq_word_statis = self._cls_word_cnt[False]

        for k, v in is_qq_word_statis.items():
            is_qq_numer_word_cnt = self._session.query(ClassWordCount).filter_by(cls='is_qq_number', word=k).one()
            if is_qq_numer_word_cnt is None:
                continue

            is_qq_numer_word_cnt = is_qq_numer_word_cnt.cnt
            is_qq_word_statis[k] = v + is_qq_numer_word_cnt


        for k, v in is_not_qq_word_statis.items():
            is_not_qq_numer_word_cnt = self._session.query(ClassWordCount).filter_by(cls='is_not_qq_number', word=k).one()
            if is_not_qq_numer_word_cnt is None:
                continue

            is_not_qq_numer_word_cnt = is_not_qq_numer_word_cnt.cnt
            is_qq_word_statis[k] = v + is_not_qq_numer_word_cnt


        self.__save()



    def __records(self):
        for idx, line in enumerate(self._data_file):
            fields = line.split('\t')
            if len(fields) != 2:
                print('format error, line = %d, content = %s' % (idx, line))
                continue

            sms = fields[0]
            qq = fields[1]

            for ctx, number in self.__get_numbers(sms):
                ctx = jieba.cut(ctx)
                filtered_ctx = filter(lambda x : x not in self._stop_wrods, ctx)

                if number == qq:
                    yield  filtered_ctx, True
                else:
                    yield  filtered_ctx, False


    def __get_numbers(self, sms):
        while True:
            so = re.search('[0-9]+', sms)
            if so is None:
                break

            start = so.span()[0]
            end = so.span()[1]
            ctx_start = start - self._ctx_len

            if ctx_start < 0:
                ctx = sms[:start]
            else:
                ctx = sms[ctx_start:start]

            yield ctx, so.group()

            sms = sms[end:]
            if not sms:
                break


    def __get_word_count(self, w):
        word_cnt = list(self._session.query(ClassWordCount).filter_by(word=w))
        if word_cnt is None:
            return 1, 1

        if len(word_cnt) != 2:
            print('ERROR: %s cls is %d ' % (w, len(word_cnt)))


        if word_cnt[0].cls == 'is_qq_number' and word_cnt[1].cls == 'is_not_qq_number':
            is_qq_word_cnt = word_cnt[0].cnt
            is_not_qq_word_cnt = word_cnt[1].cnt
        elif word_cnt[1].cls == 'is_qq_number' and word_cnt[0].cls == 'is_not_qq_number':
            is_qq_word_cnt = word_cnt[1].cnt
            is_not_qq_word_cnt = word_cnt[0].cnt
        else:
            print("ERROR:  word_cnt[0].cls = %s,  word_cnt[1].cls = %s" % ( word_cnt[0].cls,  word_cnt[1].cls))
            is_qq_word_cnt = 1
            is_not_qq_word_cnt = 1

        return is_qq_word_cnt, is_not_qq_word_cnt


    def __get_cls_count(self, cls):
        cls_count = self._session.query(ClassCount).filter_by(cls=cls).one()
        return cls_count.cnt


    def predict(self, ctx):
        is_qq_cls_cnt = self.__get_cls_count('is_qq_number')
        is_not_qq_cls_cnt = self.__get_cls_count('is_not_qq_number')
        total_cnt = is_qq_cls_cnt + is_not_qq_cls_cnt

        p_positive = is_qq_cls_cnt / total_cnt
        p_negative = is_not_qq_cls_cnt / total_cnt

        for w in ctx:
            is_qq_cls_word_cnt, is_not_qq_cls_word_cnt = self.__get_word_count(w)
            p_positive = p_positive * math.log2(is_qq_cls_word_cnt / is_qq_cls_cnt)
            p_negative = p_negative * math.log(is_not_qq_cls_cnt / is_not_qq_cls_word_cnt)

        return True if p_positive > p_negative else False