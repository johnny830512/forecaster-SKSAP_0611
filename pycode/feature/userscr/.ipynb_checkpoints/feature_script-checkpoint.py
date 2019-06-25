# coding: utf8
import pandas as pd
import numpy as np
import datetime as dt
from .template import FeatureTransform
from .template import Utility_Method
from sqlalchemy import create_engine
from pycode.lims import lims

class SAP_High_CRC_V2(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[0]
        self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()

    def _get_half_lims(self, time):
        start_time = time - dt.timedelta(days=3)
        lims_cols = ['component_name', 'sampled_date', 'result_value', 'gradename']
        items = ['茶袋保持力', '假比重', '平均粒徑μm', '膠體強度', '含水率']  # 半成品
        high_grade = ['BC283HA', 'BC283FHA', 'BC383GA', 'BC8000']  # 高中和度
        half_lims = self.lims.get_lims(lims_cols, ['SKS_T370'], items, high_grade, start_time, time)
        
        display(half_lims.tail())
        half_lims = half_lims.interpolate()
        display(half_lims.tail())
        
        return half_lims

    def _get_data(self, time):
        # overwriting
        start_time = time - dt.timedelta(days=2)

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['SSAP-TD385F.PV', 'SSAP-TD385A1.PV', 'SSAP-DPF389.PV', 'SSAP-FB389.PV', 'SSAP-PVU01.PV',
                       'half_假比重', 'half_平均粒徑μm', 'half_茶袋保持力', 'EC', 'FT_total', 'TD385A']
        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df


class SAP_High_AAP7_V2(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[1]
        self.scaler = self.scaler_dict[self.predict_name]
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
        feature_col = ['SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-PD3850.PV', 'SSAP-PD3851.PV', 'SSAP-TD385C.PV',
                       'SSAP-TD385D.PV', 'SSAP-TD385F.PV', 'SSAP-MD385A.PV', 'SSAP-PD385A.PV', 'SSAP-TD385A1.PV',
                       'SSAP-DPF389.PV', 'SSAP-FB389.PV', 'SSAP-SSJ384.PV', 'half_假比重', 'half_平均粒徑μm',
                       'SSAP-FJ3021.PV', 'EC', 'bad_ratio', 'FT_total', 'TD385A']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df

class SAP_High_AAP3(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[2]
        self.scaler = self.scaler_dict[self.predict_name]
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
        feature_col = ['SSAP-FB385.PV', 'SSAP-TD3851.PV', 'SSAP-TD385A5.PV', 'SSAP-TD385A8.PV', 'SSAP-DPF389.PV',
                       'SSAP-MJ384.PV', 'SSAP-TTF389.PV', 'half_假比重', 'EC-SAP', 'mix', 'FT_total']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df[feature_col]

class SAP_High_AAP3_V2(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[3]
        self.scaler = self.scaler_dict[self.predict_name]
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
        feature_col = ['SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-TD3851.PV', 'SSAP-MD385A.PV', 'SSAP-TD385A1.PV',
                       'SSAP-PVU01.PV', 'half_假比重', 'half_茶袋保持力', 'EC-SAP', 'bad_ratio', 'mix', 'FT_total',
                       'TD385A', 'SSAP-PD3850.PV', 'SSAP-TD385F.PV']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df[feature_col]

class SKSAP_high_crc_v3(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[4]
        self.scaler = self.scaler_dict[self.predict_name]
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
        feature_col = ['SSAP-MD386.PV', 'SSAP-FB385.PV', 'SSAP-PD3850.PV', 'SSAP-TD3851.PV',
            'SSAP-TD385A.PV', 'SSAP-TD385F.PV', 'SSAP-PD385A.PV', 'SSAP-TD385A1.PV',
            'SSAP-TD385A8.PV', 'SSAP-DPF389.PV', 'SSAP-FB389.PV', 'SSAP-MJ384.PV',
            'SSAP-PVU01.PV', 'SSAP-SSJ384.PV', 'SSAP-TTF389.PV', 'half_假比重',
            'half_平均粒徑μm', 'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP', 'EC',
            'bad_ratio', 'mix', 'bridge', 'FT_total', 'TD385', 'TD385A', 'PD385A']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_high_3psi_v3(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[5]
        self.scaler = self.scaler_dict[self.predict_name]
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
        feature_col = ['SSAP-MD386.PV', 'SSAP-FB385.PV', 'SSAP-PD3850.PV', 'SSAP-TD3851.PV',
            'SSAP-TD385A.PV', 'SSAP-TD385F.PV', 'SSAP-PD385A.PV', 'SSAP-TD385A1.PV',
            'SSAP-TD385A8.PV', 'SSAP-DPF389.PV', 'SSAP-FB389.PV', 'SSAP-MJ384.PV',
            'SSAP-PVU01.PV', 'SSAP-SSJ384.PV', 'SSAP-TTF389.PV', 'SSAP-WV383.PV',
            'half_假比重', 'half_平均粒徑μm', 'half_茶袋保持力', 'SSAP-FJ3021.PV', 'EC-SAP',
            'EC', 'bad_ratio', 'mix', 'bridge', 'FT_total', 'TD385', 'TD385A',
            'PD385A']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
    
class SKSAP_high_7psi_v3(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[6]
        self.scaler = self.scaler_dict[self.predict_name]
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
        feature_col = ['SSAP-FB385.PV', 'SSAP-MD385.PV', 'SSAP-PD3850.PV', 'SSAP-PD3851.PV',
            'SSAP-TD3851.PV', 'SSAP-TD3853.PV', 'SSAP-TD3854.PV', 'SSAP-TD385A.PV',
            'SSAP-TD385B.PV', 'SSAP-TD385C.PV', 'SSAP-TD385D.PV', 'SSAP-TD385E.PV',
            'SSAP-TD385F.PV', 'SSAP-MD385A.PV', 'SSAP-PD385A.PV', 'SSAP-PD385A1.PV',
            'SSAP-PD385A2.PV', 'SSAP-TD385A1.PV', 'SSAP-TD385A13.PV',
            'SSAP-TD385A14.PV', 'SSAP-TD385A2.PV', 'SSAP-DPF389.PV',
            'SSAP-FB389.PV', 'SSAP-FIP382.PV', 'SSAP-FJ3843.PV', 'SSAP-SSJ384.PV',
            'SSAP-WT383.PV', 'SSAP-WV383.PV', 'half_假比重', 'half_平均粒徑μm',
            'half_茶袋保持力', 'SSAP-FJ3021.PV', 'SSAP-FT1170.PV', 'SSAP-FT1173.PV',
            'SSAP-FT3161.PV', 'SSAP-FT3162.PV', 'SSAP-WV317A.PV', 'SSAP-WV317B.PV',
            'SSAP-FJ3011.PV', 'SSAP-FJ3022.PV', 'SSAP-FJ303A.PV', 'EC', 'bad_ratio',
            'mix', 'bridge', 'FT_total']

        half_lims_df = self._get_half_lims(time)
        df = self.u_method.combine_data(self.rtpms, half_lims_df, time)
        df = self.u_method.create_feature(df)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
    ###################################以下為新版
    ###################################以下為新版
    ###################################以下為新版
    ###################################以下為新版
    ###################################以下為新版
class SKSAP_CRC_DNN_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[7]
        self.scaler = self.scaler_dict[self.predict_name]
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
        
        df = self.u_method.add_grade(df) #如果要加入品別
        df = self.u_method.add_last_value(df,target_col = "茶袋保持力" , new_colname = "last_tea") #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_CRC_RF_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[8]
        self.scaler = self.scaler_dict[self.predict_name]
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
        
        df = self.u_method.add_grade(df) #如果要加入品別
        df = self.u_method.add_last_value(df,target_col = "茶袋保持力" , new_colname = "last_tea") #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 
    
class SKSAP_CRC_XGB_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[9]
        self.scaler = self.scaler_dict[self.predict_name]
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
        
        df = self.u_method.add_grade(df) #如果要加入品別
        df = self.u_method.add_last_value(df,target_col = "茶袋保持力" , new_colname = "last_tea") #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_aap3_RF_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[10]
        self.scaler = self.scaler_dict[self.predict_name]
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
        
        df = self.u_method.add_grade(df) #如果要加入品別
        #df = self.u_method.add_last_value(df,target_col = "茶袋保持力" , new_colname = "last_psi3") #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_aap7_DNN_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[11]
        self.scaler = self.scaler_dict[self.predict_name]
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
        
        df = self.u_method.add_grade(df) #如果要加入品別
        df = self.u_method.add_last_value(df,target_col = '0.7 psi AAP' , new_colname = "last_psi7") #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
class SKSAP_aap7_RF_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[12]
        self.scaler = self.scaler_dict[self.predict_name]
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
        
        df = self.u_method.add_grade(df) #如果要加入品別
        #df = self.u_method.add_last_value(df,target_col = '0.7 psi AAP' , new_colname = "last_psi7") #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 
class SKSAP_aap7_XGB_0605(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[13]
        self.scaler = self.scaler_dict[self.predict_name]
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
        
        df = self.u_method.add_grade(df) #如果要加入品別
        #df = self.u_method.add_last_value(df,target_col = '0.7 psi AAP' , new_colname = "last_psi7") #如果要加入上筆LIMS數據
        
        
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        data_s = self.scaler.transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df 