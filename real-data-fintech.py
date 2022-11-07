import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from functools import partial
from fairdata import FairData

datafile = 'data/Fintech/processed_data.npz'
M = 50


if __name__ == '__main__':
    if os.path.exists(datafile):
        data = np.load(datafile)
        a = data['arr_0']
        s = data['arr_1']
        y = data['arr_2']
    else:
        df_raw_1 = pd.read_excel('data/Fintech/Fintech-fairnessJun2020.xlsx', sheet_name='Approved&Default')
        df_raw_2 = pd.read_excel('data/Fintech/Fintech-fairnessJun2020.xlsx', sheet_name='Rejected')
        df_raw_1['approved_dum'] = 1
        df_raw_2['approved_dum'] = 0
        df_tmp = pd.concat([
            df_raw_1.drop(['loan_transferred_date', 'def_flag'], axis=1),
            df_raw_2.rename(columns={'loan_request_initial_id': 'loan_request_id'}),
        ]).rename(columns={
            'noofconnections': 'connections',
            'noofapps': 'apps',
            'noofsms': 'sms',
            'noofcontacts': 'contacts',
        }).astype({
            'customer_id': 'Int64',
            'loan_request_id': 'Int64',
            'age': 'float',
            'connections': 'float',
            'apps': 'float',
            'sms': 'float',
            'contacts': 'float',
            'approved_dum': 'Int64',
        })
        df_raw_3 = pd.read_csv('data/Fintech/Cashe_information.csv').rename(columns={
            'AGE': 'age',
        }).astype({
            'customer_id': 'Int64',
            'loan_request_id': 'Int64',
            'age': 'float',
            'connections': 'float',
            'apps': 'float',
            'sms': 'float',
            'contacts': 'float',
            'approved_dum': 'Int64',
        })
        df_raw = pd.merge(
            df_tmp.dropna(),
            df_raw_3.loc[:, ['loan_request_id', 'salary', 'loan_amount', 'CIBIL']].dropna(), 
            how='left', on='loan_request_id').dropna()

        df = df_raw[~df_raw.gender.isna()]
        df['dum'] = 0
        df.loc[(df.age < 28) & (df.gender != 'f'), 'dum'] = 1
        df.loc[(df.age >= 28) & (df.gender == 'f'), 'dum'] = 2
        df.loc[(df.age >= 28) & (df.gender != 'f'), 'dum'] = 3

        df = df.drop(['gender', 'age', 'customer_id', 'loan_request_id', 'CIBIL', 'loan_amount'], axis=1).astype({'approved_dum': 'int64'})

        log_vars = ['salary', 'connections',  'apps', 'sms', 'contacts']
        for c in log_vars:
            df[c] = np.log(df[c] + 1)
        norm_vars = log_vars
        scaler = StandardScaler().fit(df[norm_vars])
        df[norm_vars] = scaler.transform(df[norm_vars])


        y = df.approved_dum.values.reshape(-1, 1)
        s = df.dum.values.reshape(-1, 1)
        a = df.drop(['approved_dum', 'dum'], axis=1).values

        np.savez('data/Fintech/processed_data.npz', a, s, y)


    def compute(i):
        resfile = f'data/Fintech/res/{i}.npy'
        if os.path.exists(resfile):
            return np.load(resfile)
        a_train, a_test, s_train, s_test, y_train, y_test = train_test_split(a, s, y, test_size=10000, random_state=i)
        fairdata_ortho = FairData(s_train, a_train, y_train, preprocess_method='o')
        fairdata_mdm = FairData(s_train, a_train, y_train, preprocess_method='m')
        fairdata_mdm_eval = fairdata_mdm.evaluate(
            a_test, s_test, y_test, metrics=['cf', 'cfbm', 'acc'], p_range=0.05, b=50
        )
        fairdata_ortho_eval = fairdata_ortho.evaluate(
            a_test, s_test, y_test, metrics=['cf', 'cfbm', 'acc'], methods=['FLAP-1', 'FLAP-2'], p_range=0.05, b=50
        )
        res = np.concatenate([np.array(fairdata_mdm_eval), np.array(fairdata_ortho_eval)], axis=1)
        np.save(f'data/Fintech/res/{i}.npy', res)
        return res


    pool = Pool(16)
    experiments = [pool.apply_async(compute, (i,)) for i in range(M)]
    
    res = np.array([e.get() for e in experiments])

    print(res.mean(0).round(4))
    print(res.std(0).round(4))