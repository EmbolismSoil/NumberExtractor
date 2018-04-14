import re
import jieba
from .models import *
import math
import traceback
import unicodedata
from tqdm import tqdm


class NB(object):
    def __init__(self, data_path, dict_path, stop_words_path, noise_char_path, ctx_start_len, ctx_end_len):
        self._data_path = data_path
        self._stop_wrods_path = stop_words_path
        self._ctx_start_len = ctx_start_len
        self._ctx_end_len = ctx_end_len
        self._data_file = open(data_path, 'r')
        self._stop_wrods_file = open(stop_words_path, 'r')
        self._noise_char_file = open(noise_char_path, 'r')

        jieba.set_dictionary(dict_path)
        self._stop_words = set()
        self._noise_chars = set()

        for line in self._stop_wrods_file:
            line = line.rstrip('\n')
            self._stop_words.add(line)

        for line in self._noise_char_file:
            line = line.rstrip('\n')
            self._noise_chars.add(line)

        self._stop_words = sorted(self._stop_words)

        self._cls_cnt = {False: 0, True: 0}
        self._cls_word_cnt = {False: {}, True: {}}
        self._session = DBSession()
        self._engine = engine


    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False


    def update_train(self):
        self.__do_train()
        self.__update()


    def train(self):
        self.__do_train()
        self.__save()


    def __do_train(self):
        cls_cnt = {False: 0, True: 0}
        cls_word_cnt = {False: {}, True: {}}

        for ctx, is_qq in tqdm(self.__records(self._data_file), desc='train'):
            cls_cnt[is_qq] = cls_cnt.get(is_qq, 0) + 1
            for w in ctx:
                w = w.lower()
                cls_word_cnt[is_qq][w] = cls_word_cnt[is_qq].get(w, 0) + 1

        self._cls_cnt = cls_cnt
        self._cls_word_cnt = cls_word_cnt


    def __insert_cls_cnt(self, cls, cnt):
        sql = """INSERT INTO cls_cnt VALUES('%s', %d) ON DUPLICATE KEY UPDATE cnt = cnt + %d""" % (cls, cnt, cnt)
        self._engine.execute(sql)


    def __insert_cls_word_cnt(self, cls, w, cnt):
        sql = """INSERT INTO cls_word_cnt VALUES('%s', '%s', %d) ON DUPLICATE KEY UPDATE cnt = cnt + %d""" % (cls, w, cnt, cnt)
        self._engine.execute(sql)


    def __save(self):
        self.__insert_cls_cnt('is_qq_number', self._cls_cnt[True])
        self.__insert_cls_cnt('is_not_qq_number', self._cls_cnt[False])


        is_qq_word_statis = self._cls_word_cnt[True]
        is_not_qq_word_statis = self._cls_word_cnt[False]

        for k, v in tqdm(is_qq_word_statis.items(), desc='save is_qq_word_cnt'):
            try:
                self.__insert_cls_word_cnt('is_qq_number', k, v)
            except Exception as e:
                traceback.print_exc()



        for k, v in tqdm(is_not_qq_word_statis.items(), desc='save is_not_qq_word_cnt'):
            try:
                self.__insert_cls_word_cnt('is_not_qq_number', k, v)
            except Exception as e:
                traceback.print_exc()


        #let it crash
        try:
            self._session.commit()
        except Exception as e:
            traceback.print_exc()
            self._session.rollback()


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



    def __records(self, f):
        for idx, line in enumerate(f):
            fields = line.strip('\n').split('\t')
            if len(fields) != 2:
                print('format error, line = %d, content = %s' % (idx, line))
                continue

            sms = fields[0]
            qq = fields[1]

            for ctx, number in self.__get_numbers(sms):
                if len(number) < 5 or len(number) > 11:
                    continue

                if number == qq:
                    yield  ctx, True
                else:
                    yield  ctx, False


    def __get_numbers(self, sms):
        sms = self.__filter_noise_chars(sms)

        while True:
            so = re.search('[0-9]+', sms)
            if so is None:
                break

            start = so.span()[0]
            end = so.span()[1]

            ctx1 = jieba.cut(sms[:start])
            ctx2 = jieba.cut(sms[end:])

            ctx1 = list(filter(lambda x: x not in self._stop_words and not self.is_number(x), ctx1))
            ctx2 = list(filter(lambda x : x not in self._stop_words and not self.is_number(x), ctx2))
            ctx1 = ctx1[-self._ctx_start_len:]
            ctx2 = ctx2[:self._ctx_end_len]

            ctx = ctx1 + ctx2

            yield ctx, so.group()

            sms = sms[end:]
            if not sms:
                break


    def __filter_noise_chars(self, sms):
        for w in self._noise_chars:
            sms = sms.replace(w, '')

        return sms


    def __filter_stop_wrods(self, sms):
        for w in self._stop_words:
            sms = sms.replace(w, '')

        return sms


    def __get_word_count(self, w):
        word_cnt = self._session.query(ClassWordCount).filter_by(word=w).all()
        if not word_cnt:
            return None, None

        is_qq_word_cnt = None
        is_not_qq_word_cnt = None

        for c in word_cnt:
            if c.cls == 'is_qq_number':
                is_qq_word_cnt = c.cnt
            else:
                is_not_qq_word_cnt = c.cnt

        return is_qq_word_cnt, is_not_qq_word_cnt


    def __get_cls_count(self, cls):
        cls_count = self._session.query(ClassCount).filter_by(cls=cls).one()
        return cls_count.cnt if cls_count.cnt > 0 else 1


    def predict(self, ctx):
        ctx = jieba.cut(ctx)
        ctx = filter(lambda x : x not in self._stop_words and not self.is_number(x), ctx)
        return self._predict(ctx)


    def _predict(self, ctx):
        is_qq_cls_cnt = self.__get_cls_count('is_qq_number')
        is_not_qq_cls_cnt = self.__get_cls_count('is_not_qq_number')
        total_cnt = is_qq_cls_cnt + is_not_qq_cls_cnt

        #取对数把连乘变成连加，防止乘法下溢
        p_positive = math.log(is_qq_cls_cnt / total_cnt)
        p_negative = math.log(is_not_qq_cls_cnt / total_cnt)

        for w in ctx:
            is_qq_cls_word_cnt, is_not_qq_cls_word_cnt = self.__get_word_count(w)
            #因为黑样本比较少，所以is_qq_cls_word_cnt如果是None的话，不一定说明这个词是一个小概率事件，而是因为样本没覆盖到而已
            #所以跳过这个词。如果is_not_qq_cls_word_cnt是None说明这个词在is_not_qq_cls这个分类中出现的概率很小，因为is_not_qq_cls的
            #样本数量很大，基本上都会覆盖到
            if is_qq_cls_word_cnt is None :
                continue

            if is_not_qq_cls_word_cnt is None:
                is_not_qq_cls_word_cnt = 2

            p_positive = p_positive + math.log(is_qq_cls_word_cnt / is_qq_cls_cnt)
            p_negative = p_negative + math.log(is_not_qq_cls_word_cnt / is_not_qq_cls_cnt)

        if p_positive > p_negative:
            return True
        else:
            return False


    def test(self, test_file_path):
        total_cnt = 0
        right_cnt = 0
        with open(test_file_path, 'r') as f:
            for ctx, is_qq in self.__records(f):
                ret = self._predict(ctx)
                if ret == is_qq:
                    right_cnt += 1
                else:
                    print('error: ret = %s , is_qq = %s, ctx = %s' % (str(ret), str(is_qq), str(ctx)))
                total_cnt += 1

        print('accuracy: %f' % (right_cnt / total_cnt))