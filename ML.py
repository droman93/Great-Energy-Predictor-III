import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import minmax_scale

from tqdm import tqdm
plt.style.use('ggplot')

df_BuildingMetadata = pd.read_csv('building_metadata.csv')
df_train_weather = pd.read_csv('weather_train.csv')
df_test_weather = pd.read_csv('weather_test.csv')

def weather(df_train_weather):
    df_train_weather['timestamp'] = pd.to_datetime(df_train_weather['timestamp'])
    df_train_weather['air_temperature'].fillna(value=df_train_weather['air_temperature'].mean(),inplace=True)
    df_train_weather['dew_temperature'].fillna(value=df_train_weather['dew_temperature'].median(), inplace=True)
    df_train_weather['precip_depth_1_hr'].fillna(value=df_train_weather['precip_depth_1_hr'].median(),inplace=True)
    df_train_weather['sea_level_pressure'].fillna(value=df_train_weather['sea_level_pressure'].median(),inplace=True)
    df_train_weather['wind_direction'].fillna(value=df_train_weather['wind_direction'].median(),inplace=True)
    df_train_weather['wind_speed'].fillna(value=df_train_weather['wind_speed'].median(),inplace=True)
    return df_train_weather
df_test_weather = weather(df_train_weather)

def weather(df_train_weather):
    df_train_weather['timestamp'] = pd.to_datetime(df_train_weather['timestamp'])
    df_train_weather['air_temperature'].fillna(value=df_train_weather['air_temperature'].mean(),inplace=True)
    df_train_weather['dew_temperature'].fillna(value=df_train_weather['dew_temperature'].median(), inplace=True)
    df_train_weather['precip_depth_1_hr'].fillna(value=df_train_weather['precip_depth_1_hr'].median(),inplace=True)
    df_train_weather['sea_level_pressure'].fillna(value=df_train_weather['sea_level_pressure'].median(),inplace=True)
    df_train_weather['wind_direction'].fillna(value=df_train_weather['wind_direction'].median(),inplace=True)
    df_train_weather['wind_speed'].fillna(value=df_train_weather['wind_speed'].median(),inplace=True)
    return df_train_weather
df_test_weather = weather(df_test_weather)

def BInfo(df_BuildingMetadata):
    df_BuildingMetadata.drop('year_built', axis=1, inplace=True)
    encoder = LabelEncoder()
    df_BuildingMetadata['primary_use'] = encoder.fit_transform(df_BuildingMetadata['primary_use'])
    df_BuildingMetadata['floor_count'].fillna(value=0,inplace=True)
    return df_BuildingMetadata
df_BuildingMetadata = BInfo(df_BuildingMetadata)

df_train.isna().sum()
df_train.shape

df_train.columns

''' Data from the train.csv =========================================================================='''
df_train['meter'].hist()
df_train['meter'].unique() # meter categorical data

df_train['meter_reading'].hist()

df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
df_train['month'] = df_train['timestamp'].dt.month
# df_train['week'] = df_train['timestamp'].dt.week
df_train['weekday'] = df_train['timestamp'].dt.weekday
df_train['hour'] = df_train['timestamp'].dt.hour

df_train.set_index(['building_id','timestamp'],inplace=True)
df_train.sort_index(inplace=True)

for i in range(0):
    ''' Data from the train weather=========================================================================='''
    df_train_weather.columns
    df_train_weather['timestamp'] = pd.to_datetime(df_train_weather['timestamp'])
    df_train_weather.set_index(['site_id','timestamp'],inplace=True)
    df_train_weather.isna().sum()

    # Checking air_temperature
    df_train_weather['air_temperature'].hist()
    df_train_weather['air_temperature'].isna().sum()/len(df_train_weather['air_temperature'])
    df_train_weather['air_temperature'].fillna(value=df_train_weather['air_temperature'].mean(),inplace=True)

    #
    df_train_weather['cloud_coverage'].hist()
    df_train_weather['cloud_coverage'].unique()
    df_train_weather['cloud_coverage'].fillna(value=10,inplace=True)
    df_train_weather.drop('cloud_coverage',axis=1,inplace=True)

    #
    df_train_weather['dew_temperature'].hist()
    df_train_weather['dew_temperature'].isna().sum()/len(df_train_weather['dew_temperature'])
    df_train_weather['dew_temperature'].fillna(value=df_train_weather['dew_temperature'].median(),inplace=True)

    #
    df_train_weather['precip_depth_1_hr'].hist()
    df_train_weather['precip_depth_1_hr'].isna().sum()/len(df_train_weather['precip_depth_1_hr'])
    df_train_weather['precip_depth_1_hr'].fillna(value=df_train_weather['precip_depth_1_hr'].median(),inplace=True)

    #
    df_train_weather['sea_level_pressure'].hist()
    df_train_weather['sea_level_pressure'].isna().sum()/len(df_train_weather['sea_level_pressure'])
    df_train_weather['sea_level_pressure'].fillna(value=df_train_weather['sea_level_pressure'].median(),inplace=True)

    #
    df_train_weather['wind_direction'].hist()
    df_train_weather['wind_direction'].isna().sum()/len(df_train_weather['wind_direction'])
    df_train_weather['wind_direction'].fillna(value=df_train_weather['wind_direction'].median(),inplace=True)

    df_train_weather['wind_speed'].hist()
    df_train_weather['wind_speed'].isna().sum()/len(df_train_weather['wind_speed'])
    df_train_weather['wind_speed'].fillna(value=df_train_weather['wind_speed'].median(),inplace=True)

''' Data from the BuildingMetaData=========================================================================='''
df_BuildingMetadata.columns
df_BuildingMetadata.isna().sum()
df_BuildingMetadata.set_index(['site_id','building_id'],inplace=True)

df_BuildingMetadata.set_index(['site_id','building_id'],inplace=True)
df_BuildingMetadata.boxplot()
df_BuildingMetadata[['square_feet']].boxplot()
df_BuildingMetadata['square_feet'][df_BuildingMetadata['square_feet']>=250000].count()
df_BuildingMetadata['square_feet'].isna().sum()/len(df_BuildingMetadata['square_feet'])

#
df_BuildingMetadata[['year_built']].boxplot()
df_BuildingMetadata[['year_built']].hist()
df_BuildingMetadata.groupby(df_BuildingMetadata.index.names[1]).mean()
sns.distplot(df_BuildingMetadata[['year_built']].dropna())
df_BuildingMetadata['year_built'].isna().sum()/len(df_BuildingMetadata['year_built'])
df_BuildingMetadata.drop('year_built',axis=1,inplace=True)

#
df_BuildingMetadata['primary_use'].astype('category')
df_BuildingMetadata['primary_use'].hist()
encoder = LabelEncoder()
df_BuildingMetadata['primary_use'] = encoder.fit_transform(df_BuildingMetadata['primary_use'])

#
df_BuildingMetadata['floor_count'].isna().sum()/len(df_BuildingMetadata['floor_count'])
df_BuildingMetadata['floor_count'].hist()
df_BuildingMetadata.groupby('floor_count').mean().square_feet
df_BuildingMetadata['floor_count'].fillna(value=0,inplace=True)


df_train_weather.columns
train_dataset.dtypes
train_dataset['primary_use'].hist()



scaler = StandardScaler()
scaler.fit_transform(train_dataset.select_dtypes(exclude='category').drop('meter_reading',axis=1))

X = train_dataset.values


def TestTrain(df_train,Train=True):
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    df_train['month'] = df_train['timestamp'].dt.month
    df_train['week'] = df_train['timestamp'].dt.week
    df_train['weekday'] = df_train['timestamp'].dt.weekday
    df_train['hour'] = df_train['timestamp'].dt.hour
#
    train_dataset = df_train.join(
        df_BuildingMetadata.set_index('building_id')[['site_id', 'primary_use', 'square_feet', 'floor_count']],
        on='building_id')
#
    train_dataset = train_dataset.join(df_test_weather.set_index(['timestamp', 'site_id'])[
                                           ['air_temperature', 'dew_temperature',
                                            'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
                                            'wind_speed']],
                                       on=['timestamp', 'site_id'])
#
    if Train:
        train_dataset.drop(['building_id', 'site_id', 'timestamp'], axis=1, inplace=True)
    else:
        train_dataset.drop(['building_id', 'site_id', 'timestamp', 'row_id'], axis=1, inplace=True)
#
    train_dataset['meter'] = train_dataset['meter'].astype('category')
    train_dataset['month'] = train_dataset['month'].astype('category')
    train_dataset['week'] = train_dataset['week'].astype('category')
    train_dataset['weekday'] = train_dataset['weekday'].astype('category')
    train_dataset['hour'] = train_dataset['hour'].astype('category')
    train_dataset['primary_use'] = train_dataset['primary_use'].astype('category')
    train_dataset['floor_count'] = train_dataset['floor_count'].astype('category')
    train_dataset.dropna(inplace=True)
    return train_dataset



Size_split = 50
total_pred = []
split = int(41697600/Size_split)
LoweRows = 0
UppeRows = split

for i in tqdm(range(0,Size_split)):

    cols = ['row_id', 'building_id', 'meter', 'timestamp']
    if i==0:
        df_train = pd.read_csv('test.csv', skiprows= lambda x: (x < LoweRows or x > UppeRows))
    else:
        df_train = pd.read_csv('test.csv',header=None,names=cols, skiprows= lambda x: (x < LoweRows or x > UppeRows))



    train_dataset = TestTrain(df_train,Train=False)

    X = train_dataset.values

    # cat_pred = catReg_load.predict(X)
    forest_pred = forestReg.predict(X)
    total_pred.append(forest_pred)
    print(f'Low = {LoweRows} , Up = {UppeRows} , step = {i}, dif = {len(total_pred[i])}')
    LoweRows += split
    UppeRows += split


df_train = pd.read_csv('test.csv', skiprows= lambda x: (x < 40863648 or x > 41697600))
len(total_pred)


del df_train_weather,df_train,df_BuildingMetadata
del train_dataset


from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

train_data=lgb.Dataset(X,label=y)
params = {'learning_rate':0.001}

catReg = CatBoostRegressor()
catReg.fit(X,y)
lgbReg= lgb.train(params, train_data, 100)

Size_split = 20
split = int(20216100/Size_split)
LoweRows = 0
UppeRows = split
forestReg = RandomForestRegressor(warm_start = True, n_estimators = 1)

for i in tqdm(range(Size_split)):

    cols = ['building_id','meter','timestamp','meter_reading']
    if i == 0:
        df_train = pd.read_csv('train.csv', skiprows=lambda x: (x < LoweRows or x > UppeRows))
    else:
        df_train = pd.read_csv('train.csv', header=None, names=cols, skiprows=lambda x: (x < LoweRows or x > UppeRows))

    LoweRows += split
    UppeRows += split

    train_dataset = TestTrain(df_train)
    X = train_dataset.drop('meter_reading', axis=1).values
    y = train_dataset['meter_reading'].values
    forestReg.fit(X,y)
    forestReg.n_estimators += 1



from sklearn.externals import joblib

# Output a pickle file for the model
joblib.dump(catReg, 'catReg.pkl')
joblib.dump(lgbReg, 'lgbReg.pkl')
joblib.dump(forestReg, 'forestReg.pkl')

# Load the pickle file
catReg_load = joblib.load('catReg.pkl')
lgbReg_load = joblib.load('lgbReg.pkl')
forestReg_load = joblib.load('forestReg.pkl')

del X,y
cat_pred = catReg_load.predict(X)
lgb_pred = lgbReg_load.predict(X)



forestReg.fit(X,y)


len(total_pred[49])

predicted_val = [a.tolist() for a in total_pred]
predicted_val = [y for x in predicted_val for y in x]
predicted_val = [round(x,2) for x in predicted_val]
predicted_val.append(0)
predicted_val[0]
len(predicted_val)

sample_sub = pd.read_csv('sample_submission.csv')
len(sample_sub['row_id'])
dict = {'row_id':sample_sub['row_id'],'meter_reading':predicted_val}
df_sub = pd.DataFrame(dict)
df_sub.to_csv('sub.csv',index=False)