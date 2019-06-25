"""
Author: poiroot
"""
import datetime as dt
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from ...rtpms.RTPMS import RTPMS_OleDB
from ...lims.lims import Lims



class FeatureTransform():

    def __init__(self, config):
        self.author = None
        self.config = config
        self.predict_items = list(config['predict_items'].keys())
        rtpms = create_engine(config['sql_connect']['rtpms'])
        self.rtpms = RTPMS_OleDB(rtpms)
        lims_engine = create_engine(config['sql_connect']['lims'])
        lims_server = config['lims_setting']['history_linked_server']
        lims_table = config['lims_setting']['history_view']
        self.lims = Lims(lims_engine, lims_server, lims_table)
        self.scaler_dict = dict()
        self.prep_steps_dict = dict()
        for key in self.predict_items:
            self.prep_steps_dict[key] = config['predict_items'][key]['prep_steps']
        for key in self.predict_items:
            self.scaler_dict[key] = config['predict_items'][key]['scaler']

    def _get_data(self):
        """
        Get RTPMS data

        Return type should be pd.DataFrame()
        """
        #TODO implement this method
        return pd.DataFrame()

    def transform(self, time):
        """
        Transform feature

        Return type should be pd.DataFrame()
        """
        #TODO implement this method
        return pd.DataFrame()


class Utility_Method():

    def __init__(self):
        self.tag_df = pd.read_csv('pycode/feature/userscr/tag_list.csv')
        self.range_minutes = 10
        self.rtpms_time_step = '00:01:00'

    def combine_data(self, rtpms, half_lims_df, time):

        process_max = self.tag_df['process_order'].max()
        process_min = self.tag_df['process_order'].min()
        shift_minutes = 0
        row_df = pd.DataFrame()
        half_lims_df = self.lims_pivot(half_lims_df)

        while process_max >= process_min:
            process_df = self.tag_df[self.tag_df['process_order'] == process_max]
            tag_list = process_df['tag'].tolist()
            # 判斷是否為計時的製程
            if not np.isnan(process_df['process_time'].iloc[0]):
                shift_minutes += process_df['process_time'].iloc[0]
                # 判斷是否有點位存在
                if type(tag_list[0]) is str:
                    sample_time = time - dt.timedelta(minutes=shift_minutes)
                    start_time = sample_time - dt.timedelta(minutes=self.range_minutes)
                    end_time = sample_time + dt.timedelta(minutes=self.range_minutes)
                    rtpms_df = rtpms.get_rtpms(tag_list, start_time, end_time, self.rtpms_time_step, True)
                    rtpms_df = rtpms.to_row(rtpms_df)
                    rtpms_df.reset_index(inplace=True, drop=True)
                    row_df = pd.concat([row_df, rtpms_df], axis=1)
            else:
                end_time = time - dt.timedelta(minutes=shift_minutes)
                start_time = end_time - dt.timedelta(minutes=10)
                rtpms_df = rtpms.get_rtpms(tag_list, start_time, end_time, self.rtpms_time_step)
                shift_minutes += self.get_stay_time(process_df['process'].iloc[0], rtpms_df)
                if process_df['process'].iloc[0] == 'half_tank':
                    sample_time = time - dt.timedelta(minutes=shift_minutes)
                    half_df = self.get_half_lims(sample_time, half_lims_df)
                    row_df = pd.concat([row_df, half_df], axis=1)
            process_max -= 1

        return row_df

    def lims_pivot(self, df):
        df.drop_duplicates(['SAMPLED_DATE', 'COMPONENT_NAME'], keep='first', inplace=True)
        df = df.pivot(index='SAMPLED_DATE', columns='COMPONENT_NAME', values='RESULT_VALUE')
        df.index = pd.to_datetime(df.index)
        return df

    def get_stay_time(self, process_name, df):
        if process_name == 'half_tank':
            if df['SSAP-WV383.PV'].mean() < 1000:
                return 60
            else:
                stay_time = 60 * (df['SSAP-WT370A-F'].mean() + df['SSAP-WT370B-F'].mean()) / df['SSAP-WV383.PV'].mean()
                return round(stay_time, 0)
        elif process_name == 'dope_tank':
            stay_time = 60 * ((df['SSAP-LT309.PV'].mean() + df['SSAP-LT310.PV'].mean()) / 10) / (
                        (df['SSAP-FR3111.PV'].mean() + df['SSAP-FR3112.PV'].mean()) / 1000)
            return round(stay_time, 0)
        else:
            return 0

    def get_half_lims(self, sample_time, half_lims_df):
        df = half_lims_df.loc[half_lims_df.index <= sample_time]
        if len(df) != 0:
            df.columns = ['half_' + c for c in df.columns]
            df = df.iloc[-1:]
            df.reset_index(inplace=True, drop=True)
            return df
        else:
            col = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']
            df = pd.DataFrame({'half_' + c: [np.nan] for c in col})
            return df

    # 建立feature
    def create_feature(self, df):

        # 捨去SSAP-HD385.PV欄位(無資料)、SSAP-HD385A.PV欄位(皆為0)
        df = df.drop(['SSAP-HD385A.PV'], axis=1)

        # 增加欄位EC/SAP
        df['EC-SAP'] = (df['SSAP-FIP382.PV'] + df['SSAP-FJ3843.PV']) * 0.35 / df['SSAP-WV383.PV']

        # 增加欄位總EC量
        df['EC'] = df['SSAP-FIP382.PV'] + df['SSAP-FJ3843.PV']

        # 細粉
        df['bad_ratio'] = (df['SSAP-WV317A.PV'] + df['SSAP-WV317B.PV']) / (df['SSAP-FJ303A.PV'] + df['SSAP-FJ3022.PV'])

        # 理論中和度
        df['mix'] = ((df['SSAP-FJ3011.PV'] * 0.49 * 1.5) / 40) / ((df['SSAP-FJ303A.PV'] + df['SSAP-FJ3022.PV']) / 72)

        # 架橋劑
        df['bridge'] = df['SSAP-FJ3021.PV'] / (df['SSAP-FJ303A.PV'] + df['SSAP-FJ3022.PV'])

        # 總發泡劑
        df['FT_total'] = df['SSAP-FT1170.PV'] + df['SSAP-FT1173.PV'] + df['SSAP-FT3161.PV'] + df['SSAP-FT3162.PV']

        # 改質D385軸心溫度
        df['TD385'] = (df['SSAP-TD3853.PV'] + df['SSAP-TD3854.PV']) / 2

        # 改質D385A軸心溫度
        df['TD385A'] = (df['SSAP-TD385A13.PV'] + df['SSAP-TD385A14.PV']) / 2

        # 改質D385A蒸氣壓力
        df['PD385A'] = (df['SSAP-PD385A1.PV'] + df['SSAP-PD385A2.PV']) / 2
        
        # 低發泡劑
        df['high_FT'] = df['SSAP-FT3161.PV'] + df['SSAP-FT3162.PV']

        # 高發泡劑
        df['low_FT'] = df['SSAP-FT1170.PV'] + df['SSAP-FT1173.PV']
        
        #del df['SSAP-TD3853.PV']
        #del df['SSAP-TD3854.PV']
        #del df['SSAP-TD385A13.PV']
        #del df['SSAP-TD385A14.PV']
        #del df['SSAP-PD385A1.PV']
        #del df['SSAP-PD385A2.PV']
        #del df['SSAP-FT1170.PV']
        #del df['SSAP-FT1173.PV']
        #del df['SSAP-FT3161.PV']
        #del df['SSAP-FT3162.PV']
        #del df['SSAP-FJ3011.PV']
        #del df['SSAP-FJ303A.PV']
        #del df['SSAP-FJ3022.PV']
        #del df['SSAP-WV317A.PV']
        #del df['SSAP-WV317B.PV']
        #del df['SSAP-FIP382.PV']
        #del df['SSAP-FJ3843.PV']
        #del df['SSAP-WV383.PV']
        #del df['SSAP-WT383.PV']
        return df
    def add_grade(self, df):
        
        time = dt.datetime.now()
        start_time = time - dt.timedelta(days = 100) #往回推100天找檢驗資料
        
        lims_engine = create_engine('mssql+pyodbc://sa:`1qaz2wsx@10.110.196.60/master?driver=MSSQL')
        # 設定抓取lims的起訖時間

        encode = 'utf-8'
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        lims_obj = Lims(lims_engine, 'LIMS_SKSAP', 'lims.samp_test_result_sks')
        #檢驗點
        sample_points = ['SKS_V361']
        #檢驗項目
        items = ['茶袋保持力', '0.7 psi AAP', '0.3psiAAP', '假比重']
        #品別
        grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000'] # 高中和度
        #???
        plant_unit = ['SKS_SXFU']

        # 改質高中和度
        lims_df = lims_obj.get_lims(lims_cols, sample_points, items, grade, 
                             start_time, time)
        GRADENAME = pd.DataFrame(columns = ['BC283FHA', 'BC283HA', 'BC383GA', 'BC8000'], data = [[0,0,0,0]])
        GRADENAME.reindex(columns = ['BC283FHA', 'BC283HA', 'BC383GA', 'BC8000'])
        try:
            GRADENAME[lims_df['GRADENAME'][len(lims_df)-1]] = 1  #嘗試把最近一次的品別納入訓練 
        except:
            pass #撈不到就跳過
        df = pd.concat([df,GRADENAME],axis = 1)
        return df
    def add_last_value(self, df,target_col, new_colname):
        time = dt.datetime.now()
        start_time = time - dt.timedelta(days = 100) #往回推100天找檢驗資料
        
        lims_engine = create_engine('mssql+pyodbc://sa:`1qaz2wsx@10.110.196.60/master?driver=MSSQL')
        # 設定抓取lims的起訖時間

        encode = 'utf-8'
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        lims_obj = Lims(lims_engine, 'LIMS_SKSAP', 'lims.samp_test_result_sks')
        #檢驗點
        sample_points = ['SKS_V361']
        #檢驗項目
        items = ['茶袋保持力', '0.7 psi AAP', '0.3psiAAP', '假比重']
        #品別
        grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000'] # 高中和度
        #???
        plant_unit = ['SKS_SXFU']

        # 改質高中和度
        lims_df = lims_obj.get_lims(lims_cols, sample_points, items, grade, 
                             start_time, time)
        sector = lims_df.groupby('COMPONENT_NAME')

        value = sector.get_group(target_col)
        try:
            last_value = value['RESULT_VALUE'][value.index[-1]] 
        except:
            if new_colname == "last_psi7":
                last_value = 22 #0.7 : 22
            else :
                last_value = 31 #0.3 : 31  
                            
        df[new_colname] = last_value
        return df