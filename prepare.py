import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing

from acquire import get_titanic_data, get_iris_data
from wrangle import wrangle_telco

###################### Prep Iris Data ######################

def prep_iris(cached=True):
    '''
    This function acquires and prepares the iris data from a local csv, default.
    Passing cached=False acquires fresh data from Codeup db and writes to csv.
    Returns the iris df with dummy variables encoding species.
    '''
    
    # use my aquire function to read data into a df from a csv file
    df = get_iris_data(cached)
    
    # drop and rename columns
    df = df.drop(columns='species_id').rename(columns={'species_name': 'species'})
    
    # create dummy columns for species
    species_dummies = pd.get_dummies(df.species, drop_first=True)
    
    # add dummy columns to df
    df = pd.concat([df, species_dummies], axis=1)
    
    return df

###################### Prep Titanic Data ######################

def titanic_split(df):
    '''
    This function performs split on titanic data, stratify survived.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test



def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column into
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test


def prep_titanic(cached=True):
    '''
    This function reads titanic data into a df from a csv file.
    Returns prepped train, validate, and test dfs
    '''
    # use my acquire function to read data into a df from a csv file
    df = get_titanic_data(cached)
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked using dummy columns
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns='deck')
    
    # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)
    
    # impute mean of age into null values in age column
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test

def prep_mall_data(df):
    '''
    Take the acquired mall data, does data prep, and returns
    train, test, and vaildate data splits.
    '''
    df['is_female'] = (df.gender == 'Female').astype('int')
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    return train, test, validate

###################### Prep Telco Data ######################

def telco_split(df):
    train_and_validate, test = train_test_split(df, test_size=.12, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.12, random_state=123)
    return train, validate, test

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    new_column_names = [c + '_scales' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])
    
    train = pd.concat([
        train, pd.DataFrame(scaler.transform(train[columns_to_scale]), 
                            columns=new_column_names, 
                            index=train.index),], axis=1)
    validate = pd.concat([
        validate, pd.DataFrame(scaler.transform(validate[columns_to_scale]), 
                            columns=new_column_names, 
                            index=validate.index),], axis=1)
    test = pd.concat([
        test, pd.DataFrame(scaler.transform(test[columns_to_scale]), 
                            columns=new_column_names, 
                            index=test.index),], axis=1)
    return train, validate, test

def prep_telco_data():
    df = wrangle_telco()

    train, validate, test = telco_split(df)

    scaler = sklearn.preprocessing.MinMaxScaler()

    columns_to_scale = ['monthly_charges', 'tenure', 'total_charges']
    
    train, validate, test = add_scaled_columns(train, validate, test, scaler, columns_to_scale)

    return train, validate, test