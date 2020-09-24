import pandas as pd
import numpy as np
from env import user, host, password
import os

def wrangle_telco():

#################### Acquire Telco Data ##################
    
    def get_connection(db, user=user, host=host, password=password):
        '''
        This function uses my info from my env file to
        create a connection url to access the Codeup db.
        '''
        return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    def new_telco_data():
        '''
        This function reads the telco_churn customer data from the Codeup db into a df,
        write it to a csv file, and returns the df. 
        '''
        sql_query = '''
                    SELECT customer_id, 
                    monthly_charges, 
                    tenure, 
                    total_charges
                    FROM customers
                    WHERE contract_type_id = 3;'''
        df = pd.read_sql(sql_query, get_connection('telco_churn'))
        df.to_csv('telco_churn_df.csv')
        return df

    def get_telco_data(cached=False):
        '''
        This function reads in telco churn data from Codeup database if cached == False 
        or if cached == True reads in telco_churn_df from a csv file, returns df
        '''
        if cached or os.path.isfile('telco_churn_df.csv') == False:
            df = new_telco_data()
        else:
            df = pd.read_csv('telco_churn_df.csv', index_col=0)
        return df

#################### Prepare Telco Data ##################
    
    # getting the telco data into a data frame from function
    df = get_telco_data()
    
    # replace empty space with null values
    df.total_charges = df.total_charges.replace(' ', np.nan).astype('float')
    
    # change nulls to 0
    df.total_charges = df.total_charges.fillna(0)
    
    # drop the customer_id column
    df.drop(columns=['customer_id'], inplace=True)
    
    return df