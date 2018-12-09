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
import math
import multiprocessing

class PrefixProcess(object):
    def __init__(self, prefix_w2v_df,title_w2v_df):
        self.prefix_w2v_df = prefix_w2v_df
        self.title_w2v_df = title_w2v_df

    @staticmethod
    def _prefix_in_title(item):
        '''
        计算prefix 是否在 title中，是的话返回重合的比例
        '''
        prefix = item['prefix']
        title = item['title']
        if title!= '':
            if prefix in title:
                return len(prefix)/(len(title)+1)
            return 0
        return 0

    @staticmethod
    def _levenshtein_distance(item):
        '''
        计算prefix和title的莱文斯坦距离，即编辑距离
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
        '''
        计算prefix和title的距离比例,即距离/长度
        '''
        str1 = item["prefix"]
        str2 = item["title"]
        leven_distance = item["levenshtein_distance"]

        if not isinstance(str1, str):
            str1 = "null"

        length = max(len(str1), len(str2))

        return leven_distance / (length + 5)  # 平滑
    def _get_prefix_w2v(self,item):
        '''
        计算prefix和title词向量之间的相似度
        '''
        index = item.name
        prefix = self.prefix_w2v_df.loc[index]
        title = self.title_w2v_df.loc[index]
        num = np.dot(prefix, title)
        denom = np.linalg.norm(prefix) * np.linalg.norm(title)
        return num / denom
    @staticmethod
    def _get_rp_prefix_in_title(item):
        '''
        计算title对prefix的分词之后的召回率、精确率
        '''
        prefix = item['prefix']
        title = item['title']
        
        prefix = list(jieba.cut(prefix))
        title = list(jieba.cut(title))  
        len_title = len(title)
        len_prefix = len(prefix)
        len_comm_xx = len(set(prefix) & set(title))

        recall = len_comm_xx / (len_prefix + 0.01)
        precision = len_comm_xx / (len_title + 0.01)
        acc = len_comm_xx / (len_title + len_prefix - len_comm_xx + 0.01)
        return [recall,precision,acc]
    def get_prefix_feat(self, df):
        '''
        得到所有和prefix相关的特征并保存起来
        '''
        prefix_df = pd.DataFrame()
        prefix_df = df[['prefix','title']].copy()
            
        if not os.path.exists('./cache/prefix_1.csv'): 
            prefix_df['prefix_in_title'] = prefix_df.apply(self._prefix_in_title, axis=1)
            prefix_df['levenshtein_distance'] = prefix_df.apply(self._levenshtein_distance, axis=1)
            prefix_df['distince_rate'] = prefix_df.apply(self._distince_rate, axis=1)
            save_columns = ['prefix_in_title','levenshtein_distance','distince_rate']
            prefix_df[save_columns].to_csv('./cache/prefix_1.csv')
        else:
            prefix1_df = pd.read_csv('./cache/prefix_1.csv',index_col=0)
            prefix_df = pd.concat([prefix1_df,prefix_df],axis=1)

        if not os.path.exists('./cache/prefix_2.csv'): 
            prefix_df['prefix_w2v_title'] = prefix_df.apply(self._get_prefix_w2v, axis=1)
            save_columns = ['prefix_w2v_title']
            prefix_df[save_columns].to_csv('./cache/prefix_2.csv')
        else:
            prefix2_df = pd.read_csv('./cache/prefix_2.csv',index_col=0)
            prefix_df = pd.concat([prefix2_df,prefix_df],axis=1)

        if not os.path.exists('./cache/prefix_3.csv'):
            word_level_prefix = prefix_df.apply(self._get_rp_prefix_in_title,axis=1)
            word_level_prefix = [kk for kk in word_level_prefix]
            word_level_prefix = np.array(word_level_prefix)
            prefix_df['prefix_t_recall_word'] = word_level_prefix[:,0].tolist()
            prefix_df['prefix_t_precision_word'] = word_level_prefix[:,1].tolist()
            prefix_df['prefix_t_acc_word'] = word_level_prefix[:,2].tolist()
            save_columns = ['prefix_t_recall_word','prefix_t_precision_word','prefix_t_acc_word']
            prefix_df[save_columns].to_csv('./cache/prefix_3.csv')
        else:
            prefix3_df = pd.read_csv('./cache/prefix_3.csv',index_col=0)
            prefix_df = pd.concat([prefix3_df,prefix_df],axis=1)

        #计算prefix和title的字符串长度
        prefix_df['len_prefix' ] = prefix_df.prefix.apply(lambda x:len(x))
        prefix_df['len_title' ] = prefix_df.title.apply(lambda x:len(x))
        save_columns = ['prefix_in_title','levenshtein_distance','distince_rate','prefix_w2v_title',\
                            'prefix_t_recall_word','prefix_t_precision_word','prefix_t_acc_word','len_prefix','len_title']

        save_df = prefix_df[save_columns]
            
        return save_df
class QueryProcess(object):
    def __init__(self, title_w2v_df, w2v_model):
        self.title_w2v_df = title_w2v_df
        self.w2v_model = w2v_model
    @staticmethod
    def _get_query_prediction(item):
        '''
        如果title在query中返回它对于的概率
        '''
        query = item['query_prediction']
        title = item['title']
        if query != None:
            return float(query.get(title,np.NAN))
        return np.NAN
    
    @staticmethod
    def _get_title_in_query(item):
        '''
        title是否在query中
        '''
        query = item['query_prediction']
        title = item['title']
        if query != None and title in query:
            return 1
        return 0
    
    def _get_jieba_array(self, words, size=300):
        '''
        对输入的word做结巴分词后获取对于的词向量，取平均后作为words的向量
        '''
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
        '''
        对title和query中的每一个短语计算它们之间的相似度，并
        返回最大值、平均值和加权值
        '''
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
                num = np.dot(query_cut_array, title_array)
                denom = np.linalg.norm(query_cut_array) * np.linalg.norm(title_array)
                w2v_similar = num / denom
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
    @staticmethod
    def LCS(str1, str2):
        '''
        计算str1和str2之间的最长公共子串的长度
        '''
        if str1 == '' or str2 == '':
            return 0
        len1 = len(str1)
        len2 = len(str2)

        c = np.zeros((len2+1,),dtype=np.int32)
        max_len = 0
        for i in range(len1):
            for j in range(len2-1,-1,-1):
                if str1[i] == str2[j]:
                    c[j+1] = c[j] + 1
                    if c[j+1]>=max_len:
                        max_len = c[j+1]
                else:
                    c[j+1] = 0
        return max_len

    def _get_lcs_query(self, item):
        '''
        对title和query中的每一个短语计算最长公共子串的长度，并
        返回最大值、平均值和加权值
        '''
        query_predict = item["query_prediction"]
        title = item["title"]
        item_dict = dict()
        
        if not query_predict:
            item_dict["max_lcs"] = None
            item_dict["mean_lcs"] = None
            item_dict["weight_lcs"] = None
            return item_dict
        
        lcs_list = list()
        lcs_weight_list = list()
        for key,value in query_predict.items():
            try:
                lcs_rate = self.LCS(key,title)/max(len(title),len(key))
                lcs_list.append(lcs_rate)
                lcs_weight = lcs_rate * float(value)
                lcs_weight_list.append(lcs_weight)
            except TypeError:
                continue

        if lcs_list:
            max_rate = np.nanmax(lcs_list)
            mean_rate = np.nanmean(lcs_list)
            weight_rate = np.nansum(lcs_weight_list)

            item_dict["max_rate"] = max_rate
            item_dict["mean_rate"] = mean_rate
            item_dict["weight_rate"] = weight_rate
        else:
            item_dict["max_rate"] = None
            item_dict["mean_rate"] = None
            item_dict["weight_rate"] = None

        return item_dict
    
    @staticmethod
    def _levenshtein_distance(str1, str2):
        '''
        计算str1和str2之间的莱文斯坦距离，即编辑距离
        '''

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
    def _get_leven_query(self, item):
        '''
        对title和query中的每一个短语计算编辑距离，并
        返回最大值、平均值和加权值
        '''
        query_predict = item["query_prediction"]
        title = item["title"]
        item_dict = dict()
        
        if not query_predict:
            item_dict["min_leven"] = None
            item_dict["mean_leven_rate"] = None
            item_dict["weight_leven_rate"] = None
            return item_dict
        
        distance_list = list()
        distance_rate_list = list()
        distance_weight_list = list()
        for key,value in query_predict.items():
            try:
                levenshtein_distance = self._levenshtein_distance(key,title)
                distance_rate = 1 - levenshtein_distance / max(len(key),len(title))
                disance_weight = distance_rate * float(value)
                
                distance_list.append(levenshtein_distance)
                distance_rate_list.append(distance_rate)
                distance_weight_list.append(disance_weight)
            except TypeError:
                continue

        if distance_list:
            min_leven = np.nanmin(distance_list)
            mean_leven_rate = np.nanmean(distance_rate_list)
            weight_leven_rate = np.nansum(distance_weight_list)

            item_dict["min_leven"] = min_leven
            item_dict["mean_leven_rate"] = mean_leven_rate
            item_dict["weight_leven_rate"] = weight_leven_rate
        else:
            item_dict["min_leven"] = None
            item_dict["mean_leven_rate"] = None
            item_dict["weight_leven_rate"] = None

        return item_dict
    def get_query_df(self, df):
        '''
        获取和query相关的所有特征
        '''
        query_df = pd.DataFrame()
        
        if not os.path.exists('./cache/query_1.csv'):
            query_df["query_prediction_prob"] = df.apply(self._get_query_prediction, axis=1)
            query_df["title_in_query"] = df.apply(self._get_title_in_query, axis=1)
            save_columns = ['query_prediction_prob','title_in_query']
            query_df[save_columns].to_csv('./cache/query_1.csv')
        else:
            query1_df = pd.read_csv('./cache/query_1.csv', index_col=0)
            query_df = pd.concat([query_df,query1_df],axis=1)
        
        if not os.path.exists('./cache/query_3.csv'):
            query_df["lcs_item_dict"] = df.apply(self._get_lcs_query, axis=1)
            query_df["lcs_max_rate"] = query_df["lcs_item_dict"].apply(lambda item: item.get("max_rate"))
            query_df["lcs_mean_rate"] = query_df["lcs_item_dict"].apply(lambda item: item.get("mean_rate"))
            query_df["lcs_weight_rate"] = query_df["lcs_item_dict"].apply(lambda item: item.get("weight_rate"))
            query_df = query_df.drop(columns=["lcs_item_dict"])
            save_columns = ['lcs_max_rate','lcs_mean_rate','lcs_weight_rate']
            query_df[save_columns].to_csv('./cache/query_3.csv')
        else:
            query3_df = pd.read_csv('./cache/query_3.csv', index_col=0)
            query_df = pd.concat([query_df,query3_df],axis=1)
        
        if not os.path.exists('./cache/query_4.csv'):
            query_df["leven_item_dict"] = df.apply(self._get_leven_query, axis=1)
            query_df["leven_min"] = query_df["leven_item_dict"].apply(lambda item: item.get("min_leven"))
            query_df["leven_rate_mean"] = query_df["leven_item_dict"].apply(lambda item: item.get("mean_leven_rate"))
            query_df["leven_rate_weight"] = query_df["leven_item_dict"].apply(lambda item: item.get("weight_leven_rate"))
            query_df = query_df.drop(columns=["leven_item_dict"])
            save_columns = ['leven_min','leven_rate_mean','leven_rate_weight']
            query_df[save_columns].to_csv('./cache/query_4.csv')
        else:
            query4_df = pd.read_csv('./cache/query_4.csv', index_col=0)
            query_df = pd.concat([query_df,query4_df],axis=1)
        #计算query中候选词的个数
        query_df['len_query' ] = df.query_prediction.apply(lambda x:len(x) if x!= None else 0)

        save_columns = ['query_prediction_prob','title_in_query','w2c_max_similar','w2c_mean_similar','w2c_weight_similar',\
                       'lcs_max_rate','lcs_mean_rate','lcs_weight_rate','leven_min','leven_rate_mean','leven_rate_weight','len_query']

        return query_df

class Process(object):
    
    def __init__(self):
        '''
        读取词向量文件，如果没有的话可以运行w2v.py得到，并把文件名修改成对应的
        名称，比如'train.bin'
        '''
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
        '''
        读取数据
        '''
        file_name = './DataSets/oppo_data_ronud2_20181107/data_{}.txt'.format(name)
        if name == 'test':
            columns=['prefix','query_prediction','title','tag']
            file_name = './DataSets/oppo_round2_test_B/oppo_round2_test_B.txt'
        else:
            columns=['prefix','query_prediction','title','tag','label']

        with open(file_name,'r') as f_t:
            data_lines = f_t.readlines()
        data_list = []
        for line in data_lines:
            data_list.append(line.strip().split('\t'))
        df = pd.DataFrame(data_list, columns=columns)
        del data_lines,data_list
        
        
        if name == 'test':
            df['label'] = -1
        #进行字符串的清洗
        df['label'] = df['label'].apply(lambda x: int(x))
        df["prefix"] = df["prefix"].apply(char_cleaner)
        df["title"] = df["title"].apply(char_cleaner)
        
        df["query_prediction"] = df["query_prediction"].apply(lambda x: json.loads(x) if x != '' else None)
        return df
    @staticmethod
    def _gen_predict_prefix(item):
        '''
        根据候选词预测出用户实际想输入的词汇
        '''
        prefix = item['prefix']
        query_prediction = item['query_prediction']
        if query_prediction == None:
            return prefix
        query_set =dict()
        query_list = list(query_prediction.keys())
        
        for key,value in query_prediction.items():
            for j in range(1, len(key)+1):
                if key[0:j] not in query_set:
                    query_set[key[0:j]] = float(value)
                else:
                    query_set[key[0:j]] += float(value)
        max_str = prefix
        max_str_len = 0
        max_str_value = 0
        for key, value in query_set.items():
            if value >= max_str_value:
                if len(key) >= max_str_len:
                    max_str_len = len(key)
                    max_str_value = value
                    max_str = key
        return max_str
    @staticmethod
    def _gen_ctr_feat(df, train_len):
        '''
        对所有的特征做聚类，然后获取到他们对应的转化率
        '''
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

            save_columns.extend([ctr_str,click_str,count_str])

            ctr_df = pd.merge(ctr_df, temp, on=item, how='left')
            temp.drop(item,axis=1, inplace=True)

        ctr_df[save_columns].to_csv('./cache/df_ctr_new.csv')
        return ctr_df[save_columns]

    def _get_w2v(self, data, col, size=300):
        '''
        得到当前data中的col列的词向量
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
                    #分词后做词汇的清洗
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
        '''
        根据以上的代码得到所有的特征
        '''
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
        
        #把预测的prefix替换成当前的prefix
        df['prefix'] = df.apply(self._gen_predict_prefix, axis=1)
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
        
        #去掉一些不用的数据
        df = df.drop(['prefix', 'query_prediction', 'title', 'tag'], axis = 1)
        
        #对所有的列做平均值填充并计算log特征
        for col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
        for col in df.columns:
            log_str = col + '_log'
            df[log_str] = np.log1p(df[col])

        #分割并保存数据
        train_data = df[:train_len]
        vali_data = df[train_len:train_len+vali_len]
        test_data = df[train_len+vali_len:train_len+vali_len+test_len]

        print(train_data.shape)
        print(vali_data.shape)
        print(test_data.shape)
        train_data.to_csv('./cache/train_data_new.csv')
        vali_data.to_csv('./cache/val_data_new.csv')
        test_data.to_csv('./cache/test_data_new.csv')
import time
if __name__ == "__main__":
    t0 = time.time()
    Process().gen_feat()
    print(time.time() - t0)
    del Process