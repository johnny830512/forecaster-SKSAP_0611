# coding: utf8
import pandas as pd
import numpy as np
import datetime as dt
from .template import FeatureTransform
from .template import Utility_Method
from sqlalchemy import create_engine
from ...dataport.rtpms import RTPMS_OleDB
from ...dataport.lims import Lims

class SAP_High_CRC_V2(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[0]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=3)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=2)

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-TD385F.PV', 'SSAP-TD385A1.PV', 'SSAP-DPF389.PV', 
                       'SSAP-FB389.PV', 'SSAP-PVU01.PV',
                       'half_假比重', 'half_平均粒徑μm', 'half_茶袋保持力', 'EC',
                       'FT_total', 'TD385A']
        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        #data_s = self.scaler.transform(df[feature_col])
        data_s = self.prep_steps[0].transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
    
class SKSAP_CRC_DNN_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[1]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-FD3863.PV', 'SSAP-MD386.PV', 'SSAP-TD3861.PV', 'SSAP-TD3862.PV',
               'SSAP-TD3863.PV', 'SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-PD3850.PV',
               'SSAP-PD3851.PV', 'SSAP-TD3851.PV', 'SSAP-TD385A.PV', 'SSAP-TD385B.PV',
               'SSAP-TD385C.PV', 'SSAP-TD385D.PV', 'SSAP-TD385E.PV', 'SSAP-TD385F.PV',
               'SSAP-TE385.PV', 'SSAP-MD385A.PV', 'SSAP-PD385A.PV', 'SSAP-TD385A1.PV',
               'SSAP-TD385A2.PV', 'SSAP-TD385A3.PV', 'SSAP-TD385A4.PV',
               'SSAP-TD385A5.PV', 'SSAP-TD385A6.PV', 'SSAP-TD385A7.PV',
               'SSAP-TD385A8.PV', 'SSAP-FB389.PV', 'SSAP-MJ384.PV', 'SSAP-PVU01.PV',
               'SSAP-TTF389.PV', 'SSAP-WV383.PV', 'half_假比重', 'half_含水率',
               'half_平均粒徑μm', 'half_膠體強度', 'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP',
               'EC', 'bad_ratio', 'mix', 'bridge', 'FT_total', 'high_FT', 'low_FT',
               'TD385', 'TD385A', 'PD385A', 'BC283FHA', 'BC283HA', 'BC383GA', 'BC8000',
               'last_tea']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入品別
        
        df = self.u_method.add_last_value(df,
            target_col = '茶袋保持力' , new_colname = "last_tea", 
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_CRC_RF_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[2]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-MD386.PV', 'SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-PD3850.PV',
               'SSAP-PD3851.PV', 'SSAP-TD3851.PV', 'SSAP-TD385A.PV', 'SSAP-TD385F.PV',
               'SSAP-MD385A.PV', 'SSAP-PD385A.PV', 'SSAP-TD385A1.PV',
               'SSAP-TD385A8.PV', 'SSAP-FB389.PV', 'SSAP-MJ384.PV', 'SSAP-PVU01.PV',
               'SSAP-WV383.PV', 'half_假比重', 'half_含水率', 'half_平均粒徑μm', 'half_膠體強度',
               'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC', 'bad_ratio', 'mix',
               'bridge', 'FT_total', 'high_FT', 'low_FT', 'TD385', 'TD385A', 'PD385A',
               'BC283FHA', 'BC283HA', 'BC383GA', 'BC8000', 'last_tea']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入品別
        
        df = self.u_method.add_last_value(df,
            target_col = '茶袋保持力' , new_colname = "last_tea", 
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 
    
class SKSAP_CRC_XGB_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[3]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-MD386.PV', 'SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-PD3850.PV',
               'SSAP-PD3851.PV', 'SSAP-TD3851.PV', 'SSAP-TD385A.PV', 'SSAP-TD385F.PV',
               'SSAP-MD385A.PV', 'SSAP-PD385A.PV', 'SSAP-TD385A1.PV',
               'SSAP-TD385A8.PV', 'SSAP-FB389.PV', 'SSAP-MJ384.PV', 'SSAP-PVU01.PV',
               'SSAP-WV383.PV', 'half_假比重', 'half_含水率', 'half_平均粒徑μm', 'half_膠體強度',
               'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC', 'bad_ratio', 'mix',
               'bridge', 'FT_total', 'high_FT', 'low_FT', 'TD385', 'TD385A', 'PD385A',
               'BC283FHA', 'BC283HA', 'BC383GA', 'BC8000', 'last_tea']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入品別
        
        df = self.u_method.add_last_value(df,
            target_col = '茶袋保持力' , new_colname = "last_tea", 
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入上筆LIMS數據
        

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_aap3_RF_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[4]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-FD3863.PV', 'SSAP-MD386.PV', 'SSAP-TD3861.PV', 'SSAP-TD3863.PV',
               'SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-PD3850.PV', 'SSAP-PD3851.PV',
               'SSAP-TD3851.PV', 'SSAP-TD385F.PV', 'SSAP-MD385A.PV', 'SSAP-PD385A.PV',
               'SSAP-TD385A1.PV', 'SSAP-FB389.PV', 'half_假比重', 'half_平均粒徑μm',
               'half_膠體強度', 'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC',
               'bad_ratio', 'mix', 'bridge', 'TD385', 'TD385A', 'PD385A', 'BC283FHA',
               'BC283HA', 'BC383GA', 'BC8000']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入品別
        
        #df = self.u_method.add_last_value(df,
        #    target_col = '0.3psiAAP' , new_colname = "last_psi3", 
        #    sample_points = ['SKS_V361'],
        #    grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_aap7_DNN_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[5]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-FD3863.PV', 'SSAP-TD3861.PV', 'SSAP-PD3850.PV', 'SSAP-PD3851.PV',
               'SSAP-TD3851.PV', 'SSAP-TD385A.PV', 'SSAP-TD385D.PV', 'SSAP-TD385F.PV',
               'SSAP-PD385A.PV', 'SSAP-TD385A1.PV', 'SSAP-TD385A6.PV', 'SSAP-FB389.PV',
               'SSAP-PVU01.PV', 'SSAP-WV383.PV', 'half_假比重', 'half_含水率', 'half_平均粒徑μm',
               'half_膠體強度', 'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC',
               'bad_ratio', 'mix', 'bridge', 'FT_total', 'high_FT', 'low_FT', 'TD385',
               'TD385A', 'PD385A', 'BC283FHA', 'BC283HA', 'BC383GA', 'BC8000',
               'last_psi7']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入品別 (一般品)
        
        df = self.u_method.add_last_value(df,
            target_col = '0.7 psi AAP' , new_colname = "last_psi7", 
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_aap7_RF_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[6]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-FD3863.PV', 'SSAP-TD3861.PV', 'SSAP-PD3850.PV', 'SSAP-PD3851.PV',
               'SSAP-TD3851.PV', 'SSAP-TD385A.PV', 'SSAP-TD385D.PV', 'SSAP-TD385F.PV',
               'SSAP-PD385A.PV', 'SSAP-TD385A1.PV', 'SSAP-TD385A6.PV', 'SSAP-FB389.PV',
               'SSAP-PVU01.PV', 'SSAP-WV383.PV', 'half_假比重', 'half_含水率', 'half_平均粒徑μm',
               'half_膠體強度', 'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC',
               'bad_ratio', 'mix', 'bridge', 'FT_total', 'high_FT', 'low_FT', 'TD385',
               'TD385A', 'PD385A', 'BC283FHA', 'BC283HA', 'BC383GA', 'BC8000']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入品別 (一般品)
        
        #df = self.u_method.add_last_value(df,
        #    target_col = '0.7 psi AAP' , new_colname = "last_psi7", 
        #    sample_points = ['SKS_V361'],
        #    grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 
class SKSAP_aap7_XGB_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[7]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-FD3863.PV', 'SSAP-TD3861.PV', 'SSAP-PD3850.PV', 'SSAP-PD3851.PV',
               'SSAP-TD3851.PV', 'SSAP-TD385A.PV', 'SSAP-TD385D.PV', 'SSAP-TD385F.PV',
               'SSAP-PD385A.PV', 'SSAP-TD385A1.PV', 'SSAP-TD385A6.PV', 'SSAP-FB389.PV',
               'SSAP-PVU01.PV', 'SSAP-WV383.PV', 'half_假比重', 'half_含水率', 'half_平均粒徑μm',
               'half_膠體強度', 'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC',
               'bad_ratio', 'mix', 'bridge', 'FT_total', 'high_FT', 'low_FT', 'TD385',
               'TD385A', 'PD385A', 'BC283FHA', 'BC283HA', 'BC383GA', 'BC8000']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade =  ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000'] ) 
        #如果要加入品別 (一般品)
        
        #df = self.u_method.add_last_value(df,
        #    target_col = '0.7 psi AAP' , new_colname = "last_psi7", 
        #    sample_points = ['SKS_V361'],
        #    grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']) 
        #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 
class SKSAP_normal_CRC_0715(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[8]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283FA', 'BC283FAN', 'BC283GAN', 'BC583AN', 'BC586G','BC-283UT']
        # 一般品
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-MD386.PV', 'SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-PD3850.PV',
               'SSAP-PD3851.PV', 'SSAP-TD3851.PV', 'SSAP-TD3855.PV', 'SSAP-TD385A.PV',
               'SSAP-TD385F.PV', 'SSAP-MD385A.PV', 'SSAP-PD385A.PV', 'SSAP-TD385A1.PV',
               'SSAP-TD385A8.PV', 'SSAP-FB389.PV', 'SSAP-MJ384.PV', 'SSAP-PVU01.PV',
               'SSAP-WV383.PV', 'half_假比重', 'half_含水率', 'half_平均粒徑μm', 'half_膠體強度',
               'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC', 'bad_ratio', 'mix',
               'bridge', 'FT_total', 'high_FT', 'low_FT', 'TD385', 'TD385A', 'PD385A',
               'BC283FA', 'BC283FAN', 'BC283GAN', 'BC283UT', 'BC583AN', 'BC586G',
               'last_tea']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade = ['BC283FA', 'BC283FAN', 'BC283GAN', 'BC583AN', 'BC586G','BC-283UT']) 
        #如果要加入品別 (一般品)
        
        df = self.u_method.add_last_value(df,
            target_col = "茶袋保持力" , new_colname = "last_tea", 
            sample_points = ['SKS_V361'],
            grade = ['BC283FA', 'BC283FAN', 'BC283GAN', 'BC583AN', 'BC586G','BC-283UT']) 
        #如果要加入上筆LIMS數據
        
        index = [i for i in range(len(df.columns)) if df.columns[i] == "BC-283UT"][0]
        df.columns.values[index] = "BC283UT"
        #BC-283UT 改名為 BC283UT
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 
class SKSAP_normal_aap7_0715(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[9]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283FA', 'BC283FAN', 'BC283GAN', 'BC583AN', 'BC586G','BC-283UT']
        # 一般品半成品
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-MD386.PV', 'SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-PD3850.PV',
               'SSAP-PD3851.PV', 'SSAP-TD3851.PV', 'SSAP-TD3855.PV', 'SSAP-TD385A.PV',
               'SSAP-TD385F.PV', 'SSAP-MD385A.PV', 'SSAP-PD385A.PV', 'SSAP-TD385A1.PV',
               'SSAP-TD385A8.PV', 'SSAP-FB389.PV', 'SSAP-MJ384.PV', 'SSAP-PVU01.PV',
               'SSAP-WV383.PV', 'half_假比重', 'half_含水率', 'half_平均粒徑μm', 'half_膠體強度',
               'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC', 'bad_ratio', 'mix',
               'bridge', 'FT_total', 'high_FT', 'low_FT', 'TD385', 'TD385A', 'PD385A',
               'BC283FA', 'BC283FAN', 'BC283GAN', 'BC283UT', 'BC583AN', 'BC586G',
               'last_psi7']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V361'],
            grade = ['BC283FA', 'BC283FAN', 'BC283GAN', 'BC583AN', 'BC586G','BC-283UT']) 
        #如果要加入品別 (一般品)
        
        df = self.u_method.add_last_value(df,
            target_col = '0.7 psi AAP' , new_colname = "last_psi7", 
            sample_points = ['SKS_V361'],
            grade = ['BC283FA', 'BC283FAN', 'BC283GAN', 'BC583AN', 'BC586G','BC-283UT']) 
        #如果要加入上筆LIMS數據
        
        index = [i for i in range(len(df.columns)) if df.columns[i] == "BC-283UT"][0]
        df.columns.values[index] = "BC283UT"
        #BC-283UT 改名為 BC283UT
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 
class SKSAP_normal_CRC_1_0715(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[10]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC-283UT', 'BC283FA','BC283FAN', 'BC283GAN', 'BC583AN']
        # 一期 一般品
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T170'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-FB185.PV', 'SSAP-MD1851.PV', 'SSAP-PD1850.PV', 'SSAP-PD1851.PV',
               'SSAP-TD1851.PV', 'SSAP-TD1855.PV', 'SSAP-TD185A.PV', 'SSAP-TD185E.PV',
               'SSAP-TD185F.PV', 'SSAP-MD185A.PV', 'SSAP-PD185A.PV', 'SSAP-TD185A1.PV',
               'SSAP-DPF189.PV', 'SSAP-FB189.PV', 'SSAP-PVU01.PV', 'half_假比重',
               'half_含水率', 'half_平均粒徑μm', 'half_膠體強度', 'half_茶袋保持力', 'SSAP-FJ2031.PV',
               'EC-SAP', 'EC', 'bad_ratio', 'mix', 'bridge', 'FT_total', 'high_FT',
               'low_FT', 'TD185', 'TD185A', 'PD185A', 'BC-283UT', 'BC283FA',
               'BC283FAN', 'BC283GAN', 'BC583AN', 'last_tea']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data_1(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature_1(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V161'],
            grade = ['BC-283UT', 'BC283FA','BC283FAN', 'BC283GAN', 'BC583AN']) 
        #如果要加入品別 (一般品)
        
        df = self.u_method.add_last_value(df,
            target_col = "茶袋保持力" , new_colname = "last_tea", 
            sample_points = ['SKS_V161'],
            grade = ['BC-283UT', 'BC283FA','BC283FAN', 'BC283GAN', 'BC583AN']) 
        #如果要加入上筆LIMS數據
        
        
        #index = [i for i in range(len(df.columns)) if df.columns[i] == "BC-283UT"][0]
        #df.columns.values[index] = "BC283UT"
        #BC-283UT 改名為 BC283UT
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 
class SKSAP_normal_aap7_1_0715(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[11]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=1)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC-283UT', 'BC283FA','BC283FAN', 'BC283GAN', 'BC583AN']
        # 一期 一般品
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T170'], items, high_grade, start_time, time)
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=1)
        

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-FB185.PV', 'SSAP-MD1851.PV', 'SSAP-PD1850.PV', 'SSAP-PD1851.PV',
               'SSAP-TD1851.PV', 'SSAP-TD1855.PV', 'SSAP-TD185A.PV', 'SSAP-TD185E.PV',
               'SSAP-TD185F.PV', 'SSAP-MD185A.PV', 'SSAP-PD185A.PV', 'SSAP-TD185A1.PV',
               'SSAP-DPF189.PV', 'SSAP-FB189.PV', 'SSAP-PVU01.PV', 'half_假比重',
               'half_含水率', 'half_平均粒徑μm', 'half_膠體強度', 'half_茶袋保持力', 'SSAP-FJ2031.PV',
               'EC-SAP', 'EC', 'bad_ratio', 'mix', 'bridge', 'FT_total', 'high_FT',
               'low_FT', 'TD185', 'TD185A', 'PD185A', 'BC-283UT', 'BC283FA',
               'BC283FAN', 'BC283GAN', 'BC583AN', 'last_psi7']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data_1(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature_1(df)
        
        df = self.u_method.add_grade(df,
            sample_points = ['SKS_V161'],
            grade = ['BC-283UT', 'BC283FA','BC283FAN', 'BC283GAN', 'BC583AN']) 
        #如果要加入品別 (一般品)
        
        df = self.u_method.add_last_value(df,
            target_col = "0.7 psi AAP" , new_colname = "last_psi7", 
            sample_points = ['SKS_V161'],
            grade = ['BC-283UT', 'BC283FA','BC283FAN', 'BC283GAN', 'BC583AN']) 

        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        #data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 