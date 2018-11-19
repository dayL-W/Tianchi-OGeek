import pandas as pd
import numpy as np
import os
import json
import jieba
import pickle
from tqdm import tqdm
from gensim import matutils
from utils import char_cleaner, char_list_cheaner
from gensim.models import KeyedVectors

class PrefixProcess(object):
    def __init__(self, prefix_w2v_df,title_w2v_df):
        self.prefix_w2v_df = prefix_w2v_df
        self.title_w2v_df = title_w2v_df
    
    @staticmethod
    def _get_query_prediction(item):
        '''
        计算title在query_prediction中的概率
        '''
        query = item['query_prediction']
        title = item['title']
        if query != '':
            return float(eval(query).get(title))
        return np.NAN

    @staticmethod
    def _prefix_in_title(item):
        '''
        计算prefix 是否在 title中
        '''
        prefix = item['prefix']
        title = item['title']
        if prefix in title:
            return 1
        return 0

    @staticmethod
    def _levenshtein_distance(item):
        '''
        计算莱文斯坦距离，即编辑距离
        '''
        str1 = item['prefix']
        str2 = item['title']

        if not isinstance(str1, str):
            str1 = "null"

        x_size = len(str1) + 1
        y_size = len(str2) + 1

        matrix = np.zeros((x_size, y_size), dtype=np.int_)

        for x in range(x_size):
            matrix[x, 0] = x

        for y in range(y_size):
            matrix[0, y] = y

        for x in range(1, x_size):
            for y in range(1, y_size):
                if str1[x - 1] == str2[y - 1]:
                    matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1)
                else:
                    matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1)

        return matrix[x_size - 1, y_size - 1]
    @staticmethod
    def _distince_rate(item):
        str1 = item["prefix"]
        str2 = item["title"]
        leven_distance = item["levenshtein_distance"]

        if not isinstance(str1, str):
            str1 = "null"

        length = max(len(str1), len(str2))

        return leven_distance / (length + 5)  # 平滑
    
    def _get_prefix_w2v(self, size=300):

        prefix_w2v_list = list()
        for idx, prefix in self.prefix_w2v_df.iterrows():
            if not prefix[0]:
                prefix_w2v_list.append(None)
                continue

            title = self.title_w2v_df.loc[idx]
            if not title[0]:
                prefix_w2v_list.append(None)
                continue

            similar = np.dot(prefix, title)
            prefix_w2v_list.append(similar)
        return prefix_w2v_list
   
    def get_prefix_feat(self, df):
        prefix_df = pd.DataFrame()
        prefix_df = df[['prefix','title']]
        if not os.path.exists('./cache/prefix_feat.csv'):
            prefix_df['prefix_in_title'] = prefix_df.apply(self._prefix_in_title, axis=1)
            prefix_df['levenshtein_distance'] = prefix_df.apply(self._levenshtein_distance, axis=1)
            prefix_df['distince_rate'] = prefix_df.apply(self._distince_rate, axis=1)
            prefix_df['prefix_w2v_title'] = self._get_prefix_w2v()
            
            save_df = prefix_df[['prefix_in_title','levenshtein_distance','distince_rate','prefix_w2v_title']]
            save_df.to_csv('./cache/prefix_feat.csv')
        else:
            save_df = pd.read_csv('./cache/prefix_feat.csv', index_col=0)
            
        return save_df

class QueryProcess(object):
    def __init__(self, title_w2v_df, w2v_model):
        self.title_w2v_df = title_w2v_df
        self.w2v_model = w2v_model
#         if not os.path.exists('./cache/title_w2v_dict.pkl'):
#             self.title_w2v_dict = title_w2v_df.T.to_dict(orient='list')
#             with open('./cache/title_w2v_dict.pkl','wb') as f:
#                 pickle.dump(self.title_w2v_dict, f)
#         else:
#             with open('./cache/title_w2v_dict.pkl','rb') as f:
#                 self.title_w2v_dict = pickle.load(f)
    @staticmethod
    def _get_query_prediction(item):
        '''
        计算title在query_prediction中的概率
        '''
        query = item['query_prediction']
        title = item['title']
        if query != None:
            return float(query.get(title,0))
        return np.NAN
    
    def _get_jieba_array(self, words, size=300):
        seg_cut = jieba.lcut(words)
        seg_cut = char_list_cheaner(seg_cut)

        w2v_array = list()
        for word in seg_cut:
            try:
                similar_list = self.w2v_model[word]
                w2v_array.append(similar_list)
            except KeyError:
                continue

        if not w2v_array:
            w2v_array = [None] * size
        else:
            w2v_array = matutils.unitvec(np.array(w2v_array).mean(axis=0))

        return w2v_array
    
    def _get_w2v_similar(self, item):
        item_dict = dict()

        query_predict = item["query_prediction"]

        if not query_predict:
            item_dict["max_similar"] = None
            item_dict["mean_similar"] = None
            item_dict["weight_similar"] = None
            return item_dict

        similar_list = list()
        weight_similar_list = list()

        index = item.name
        title_array = self.title_w2v_df.loc[index]

        for key, value in query_predict.items():
            query_cut_array = self._get_jieba_array(key)

            try:
                w2v_similar = np.dot(query_cut_array, title_array)
                similar_list.append(w2v_similar)
                weight_w2v_similar = w2v_similar * float(value)
                weight_similar_list.append(weight_w2v_similar)
            except TypeError:
                continue

        if similar_list:
            max_similar = np.nanmax(similar_list)
            mean_similar = np.nanmean(similar_list)
            weight_similar = np.nansum(weight_similar_list)

            item_dict["max_similar"] = max_similar
            item_dict["mean_similar"] = mean_similar
            item_dict["weight_similar"] = weight_similar
        else:
            item_dict["max_similar"] = None
            item_dict["mean_similar"] = None
            item_dict["weight_similar"] = None

        return item_dict
    def get_query_df(self, df):
        if not os.path.exists('./cache/query_feat.csv'):
            query_df = pd.DataFrame()

            query_df["query_prediction_prob"] = df.apply(self._get_query_prediction, axis=1)
            query_df["item_dict"] = df.apply(self._get_w2v_similar, axis=1)
            query_df["max_similar"] = query_df["item_dict"].apply(lambda item: item.get("max_similar"))
            query_df["mean_similar"] = query_df["item_dict"].apply(lambda item: item.get("mean_similar"))
            query_df["weight_similar"] = query_df["item_dict"].apply(lambda item: item.get("weight_similar"))
            query_df = query_df.drop(columns=["item_dict"])
            query_df.to_csv('./cache/query_feat.csv')
        else:
            query_df = pd.read_csv('./cache/query_feat.csv', index_col=0)
        return query_df

class Process(object):
    
    def __init__(self):
        if not os.path.exists('./cache/w2c_model.pkl'):
            file_name = './DataSets/merge_sgns_bigram_char300/merge_sgns_bigram_char300.txt'
            self.w2v_model = KeyedVectors.load_word2vec_format(file_name, unicode_errors="ignore")
            with open('./cache/w2c_model.pkl', 'wb') as f_t:
                pickle.dump(self.w2v_model, f_t)
        else:
            with open('./cache/w2c_model.pkl','rb') as f_t: 
                self.w2v_model = pickle.load(f_t)
    @staticmethod
    def _get_data(name):
        if name == 'test':
            columns=['prefix','query_prediction','title','tag']
        else:
            columns=['prefix','query_prediction','title','tag','label']
        file_name = './DataSets/oppo_data_ronud2_20181107/data_{}.txt'.format(name)

        with open(file_name,'r') as f_t:
            data_lines = f_t.readlines()
        data_list = []
        for line in data_lines:
            data_list.append(line.strip().split('\t'))
        df = pd.DataFrame(data_list, columns=columns)

        if name == 'test':
            df['label'] = -1
        del data_lines,data_list
        df['label'] = df['label'].apply(lambda x: int(x))
        df["prefix"] = df["prefix"].apply(char_cleaner)
        df["title"] = df["title"].apply(char_cleaner)
        df["query_prediction"] = df["query_prediction"].apply(lambda x: json.loads(x) if x != '' else None)
        return df
    @staticmethod
    def _gen_ctr_feat(df, train_len):
        ctr_df = df[['prefix','title','tag','label']]

        train_data = ctr_df[:train_len]
        item_list = [['prefix', 'title', 'tag'], ['prefix', 'title'], ['prefix', 'tag'], ['title', 'tag'],["prefix"], ["title"], ["tag"]]

        save_columns = []
        for item in item_list:
            ctr_str = '_'.join(item)+'_ctr'
            click_str = '_'.join(item)+'_click'
            count_str = '_'.join(item)+'_count'

            temp = train_data.groupby(item, as_index = False)['label'].agg({click_str:'sum', count_str:'count'})
            temp[ctr_str] = temp[click_str]/(temp[count_str] + 5)
            temp[ctr_str+'log'] = np.log1p(temp[ctr_str])
            temp[click_str+'log'] = np.log1p(temp[click_str])
            temp[count_str+'log'] = np.log1p(temp[count_str])

            save_columns.extend([ctr_str,ctr_str+'log',click_str,click_str+'log',count_str,count_str+'log'])

            ctr_df = pd.merge(ctr_df, temp, on=item, how='left')
            temp.drop(item,axis=1, inplace=True)
            for col in temp.columns:
                ctr_df[col].fillna(temp[col].mean(), inplace=True)
        ctr_df[save_columns].to_csv('./cache/df_ctr_new.csv')
        return ctr_df[save_columns]

    def _get_w2v(self, data, col, size=300):
        '''
        得到所需要特征的词向量
        '''
        file_name = './cache/{col}_w2v_df.csv'.format(col=col)
        columns = ['{}_w2v_{}'.format(col, i) for i in range(size)]

        with open(file_name, 'a', encoding='utf-8') as f:
            # write columns
            f.write(','.join(columns) + '\n')
            for idx, item in data[col].items():
                if item == 'null':
                    item_list = [''] * size
                elif not item:
                    item_list = [''] * size
                else:
                    seg_cut = jieba.lcut(item)
                    seg_cut = char_list_cheaner(seg_cut)

                    w2v_array = list()
                    for word in seg_cut:
                        try:
                            similar_list = self.w2v_model[word]
                            w2v_array.append(similar_list)
                        except KeyError:
                            pass

                    if not w2v_array:
                        item_list = [''] * size
                    else:
                        #取词向量的平均值
                        item_list = matutils.unitvec(np.array(w2v_array,dtype=np.float32).mean(axis=0))
                f.write(','.join(map(str, item_list)) + '\n')

    def gen_feat(self):
        train_data = self._get_data(name='train')
        vali_data = self._get_data(name='vali')
        test_data = self._get_data(name='test')
        
        print('train: ',train_data.shape)
        print('test: ',test_data.shape)
        print('vali: ',vali_data.shape)
        
        train_len = train_data.shape[0]
        vali_len = vali_data.shape[0]
        test_len = test_data.shape[0]

        df = pd.concat([train_data, vali_data, test_data], axis=0, ignore_index=True)
        
        
        #CTR
        print('gen ctr')
        if not os.path.exists('./cache/df_ctr_new.csv'):
            ctr_df = self._gen_ctr_feat(df, train_len)
        else:
            ctr_df = pd.read_csv('./cache/df_ctr_new.csv', index_col=0, dtype=np.float32)
        df = pd.concat([df,ctr_df],axis=1)
        del ctr_df
        
        #prefix and title
        print('gen w2v')
        if not os.path.exists('./cache/prefix_w2v_df.csv'):
            self._get_w2v(df,'prefix')
        if not os.path.exists('./cache/title_w2v_df.csv'):
            self._get_w2v(df,'title')
        prefix_w2v_df = pd.read_csv('./cache/prefix_w2v_df.csv', header=0,dtype=np.float32)
        title_w2v_df = pd.read_csv('./cache/title_w2v_df.csv', header=0,dtype=np.float32)
        print('prefix_w2v',prefix_w2v_df.shape)
        print('title_w2v',title_w2v_df.shape)
        
        print('gen prefix')
        prefix_process = PrefixProcess(prefix_w2v_df,title_w2v_df)
        prefix_df = prefix_process.get_prefix_feat(df)
        df = pd.concat([df,prefix_df],axis=1)
        del prefix_df
        
        print('gen query')
        query_process = QueryProcess(title_w2v_df, self.w2v_model)
        query_df = query_process.get_query_df(df)
        df = pd.concat([df,query_df],axis=1)
        del query_df
        
        df = df.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
 
        train_data = df[:train_len]
        vali_data = df[train_len:train_len+vali_len]
        test_data = df[train_len+vali_len:train_len+vali_len+test_len]

        print(train_data.shape)
        print(vali_data.shape)
        print(test_data.shape)
        train_data.to_csv('./cache/train_data_new.csv')
        vali_data.to_csv('./cache/val_data_new.csv')
        test_data.to_csv('./cache/test_data_new.csv')
if __name__ == "__main__":
    t0 = time.time()
    Process().gen_feat()
    print(time.time() - t0)