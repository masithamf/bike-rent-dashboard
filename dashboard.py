import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_casual_cust_df(df):
    sum_casual_cust_df = df["casual"].sum()
    return sum_casual_cust_df

def create_registered_cust_df(df):
    sum_registered_cust_df = df["registered"].sum()
    return sum_registered_cust_df

def create_total_cust_df(df):
    sum_total_cust_df = df["cnt"].sum()
    return sum_total_cust_df

def create_hourly_stat_df(df):
    hourly_stat_df = df[['hr', 'cnt']].rename(columns={
    'hr': 'Hour',
    'cnt': 'User'
    }) 
    hourly_stat_df.sort_values(by='Hour', inplace=True)
    return hourly_stat_df

def create_daily_stat_df(df):
    daily_stat_df = df[['weekday', 'cnt']].rename(columns={
    'weekday': 'Day',
    'cnt': 'User'
    }) 
    daily_stat_df.sort_values(by='Day', inplace=True)
    return daily_stat_df

def create_monthly_stat_df(df):
    monthly_stat_df = df[['mnth', 'cnt']].rename(columns={
    'mnth': 'Month',
    'cnt': 'User'
    }) 
    monthly_stat_df.sort_values(by='Month', inplace=True)
    return monthly_stat_df

def create_user_byseason_df(df):
    user_byseason_df = df[['season', 'cnt']].rename(columns={
    'season': 'Season',
    'cnt': 'User'
    })
    user_byseason_df = user_byseason_df.sort_values(by='Season')
    return user_byseason_df

def create_user_byweather_df(df):
    user_byweather_df = df[['weathersit', 'cnt']].rename(columns={
    'weathersit': 'Weather',
    'cnt': 'User'
    })
    user_byweather_df = user_byweather_df.sort_values(by='Weather')
    return user_byweather_df

def create_user_byftemp_df(df, bins=15):
    byftemp_df = df[['atemp', 'cnt']].rename(columns={
    'atemp': 'FTemp',
    'cnt': 'User'
    })
    byftemp_df['ftemp_grouped'] = pd.cut(byftemp_df['FTemp'], bins=bins)
    user_byftemp_df = byftemp_df.groupby('ftemp_grouped')['User'].sum().reset_index()
    return user_byftemp_df

def create_user_bytemp_df(df, bins=15):
    bytemp_df = df[['temp', 'cnt']].rename(columns={
    'temp': 'Temp',
    'cnt': 'User'
    })
    bytemp_df['temp_grouped'] = pd.cut(bytemp_df['Temp'], bins=bins)
    user_bytemp_df = bytemp_df.groupby('temp_grouped')['User'].sum().reset_index()
    return user_bytemp_df

def create_user_byhum_df(df, bins=15):
    byhum_df = df[['hum', 'cnt']].rename(columns={
    'hum': 'Hum',
    'cnt': 'User'
    })
    byhum_df['hum_grouped'] = pd.cut(byhum_df['Hum'], bins=bins)
    user_byhum_df = byhum_df.groupby('hum_grouped')['User'].sum().reset_index()
    return user_byhum_df

def create_user_bywindspeed_df(df, bins=15):
    bywindspeed_df = df[['windspeed', 'cnt']].rename(columns={
    'windspeed': 'WindSpeed',
    'cnt': 'User'
    })
    bywindspeed_df['WS_grouped'] = pd.cut(bywindspeed_df['WindSpeed'], bins=bins)
    user_bywindspeed_df = bywindspeed_df.groupby('WS_grouped')['User'].sum().reset_index()
    return user_bywindspeed_df

# Loading Data
day_df = pd.read_csv("day_data.csv")    
hour_df = pd.read_csv("hour_data.csv")  

# Convert to Datetime
datetime_column_day = ["dteday"]
for column in datetime_column_day:
  day_df[column] = pd.to_datetime(day_df[column])

datetime_column_hour = ["dteday"]
for column in datetime_column_hour:
  hour_df[column] = pd.to_datetime(hour_df[column])


# Filter by Date Range
min_date = day_df["dteday"].min()
max_date = day_df["dteday"].max()

with st.sidebar:
    st.image("cragoo.png") #mengambil image dari local
    start_date, end_date = st.date_input(
        label='Sharing Date',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
        format="MM.DD.YYYY",
    )

main_df = day_df[(day_df["dteday"] >= str(start_date)) & 
                (day_df["dteday"] <= str(end_date))]

sum_casual_cust_df = create_casual_cust_df(main_df)
sum_registered_cust_df = create_registered_cust_df(main_df)
sum_total_cust_df = create_total_cust_df(main_df)
hourly_stat_df = create_hourly_stat_df(hour_df)
daily_stat_df = create_daily_stat_df(main_df)
monthly_stat_df = create_monthly_stat_df(main_df)
user_byseason_df = create_user_byseason_df(main_df)
user_byweather_df = create_user_byweather_df(main_df)
user_byftemp_df = create_user_byftemp_df(main_df)
user_bytemp_df = create_user_bytemp_df(main_df)
user_byhum_df = create_user_byhum_df(main_df)
user_bywindspeed_df = create_user_bywindspeed_df(main_df)

# Visualisasi Dashboard
st.header(':violet[CRAGOO Bike Rental Dashboard] :sunflower:')

# Subheader1
st.subheader('Bike rental Statistics :chart_with_upwards_trend:')

# Metric Pelanggan
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Casual Customer", value=sum_casual_cust_df)

with col2:
    st.metric("Registered Customer", value=sum_registered_cust_df)

with col3:
    st.metric("Total Customer", value=sum_total_cust_df)

# Statistik Distribusi Pelanggan
col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(hourly_stat_df["Hour"], hourly_stat_df["User"], color="#dd86f7")
    ax.set_title("Hourly Statistic", loc="center", fontsize=100)
    ax.set_ylabel("User",fontsize=100)
    ax.set_xlabel("Hours in a Day", fontsize=100)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(daily_stat_df["Day"], daily_stat_df["User"], color="#dd86f7")
    ax.set_title("Weekly Statistic", loc="center", fontsize=100)
    ax.set_ylabel("User",fontsize=100)
    ax.set_xlabel("Days in a Week",fontsize=100)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(monthly_stat_df["Month"], monthly_stat_df["User"], color="#dd86f7")
    ax.set_title("Monthly Statistic", loc="center", fontsize=100)
    ax.set_ylabel("User",fontsize=100)
    ax.set_xlabel("Months",fontsize=100)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)

# Subheader2
st.subheader('Our Casual and Registered Customer Sparsity :sparkles:')

def scatterplot_registered_vs_casual(df):
    plt.title("Scatter Plot: Registered vs Casual Users", fontsize=100)
    plt.figure(figsize=(15, 5))
    sns.scatterplot(data=df, x='registered', y='casual', hue='weekday', palette='bright')
    plt.legend(title='Weekday', loc='upper left')
    plt.grid(True)
    st.pyplot(plt)
scatterplot_registered_vs_casual(day_df)

# Subheader3
st.subheader('How Environmental and Seasonal Setting Affect Customer Behavior :sunny:')

# Barchart Distribusi Pelanggan
col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(user_byseason_df["Season"], user_byseason_df["User"], color="#dd86f7")
    ax.set_title("Number of Customer by Season", loc="center", fontsize=100)
    ax.set_ylabel("Customer",fontsize=100)
    ax.set_xlabel("Season",fontsize=100)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)

    st.caption("1=springer, 2=summer, 3=fall, 4=winter")

with col2:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(user_byweather_df["Weather"], user_byweather_df["User"], color="#dd86f7")
    ax.set_title("Number of Customer by Weather", loc="center", fontsize=100)
    ax.set_ylabel("Customer",fontsize=100)
    ax.set_xlabel("Weather",fontsize=100)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)

    st.caption("1: Clear, Partly cloudy. 2: Mist + Few clouds. 3: Light Snow, Light Rain + Thunderstorm. 4: Heavy Rain + Ice Pallets + Thunderstorm + Fog")

with col3:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(user_byftemp_df["ftemp_grouped"].astype(str), user_byftemp_df["User"], color="#dd86f7")
    ax.set_title("Number of Customer by Feeling Temperature", loc="center", fontsize=100)
    ax.set_ylabel("Customer",fontsize=100)
    ax.set_xlabel("Feeling Temperature in Celcius",fontsize=100)
    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)

    st.caption("Feeling temperature has been divided to 50")

col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(user_bytemp_df["temp_grouped"].astype(str), user_bytemp_df["User"], color="#dd86f7")
    ax.set_title("Number of Customer by Temperature", loc="center", fontsize=100)
    ax.set_ylabel("Customer",fontsize=100)
    ax.set_xlabel("Temperature in Celcius",fontsize=100)
    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)

    st.caption("Temperature has been divided to 41")

with col2:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(user_byhum_df["hum_grouped"].astype(str), user_byhum_df["User"], color="#dd86f7")
    ax.set_title("Number of Customer by Humidity", loc="center", fontsize=100)
    ax.set_ylabel("Customer",fontsize=100)
    ax.set_xlabel("Humidity",fontsize=100)
    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)
    
    st.caption("Humidity has been divided to 100")

with col3:
    fig, ax = plt.subplots(figsize=(50,25))
    ax.bar(user_bywindspeed_df["WS_grouped"].astype(str), user_bywindspeed_df["User"], color="#dd86f7")
    ax.set_title("Number of Customer by Windspeed", loc="center", fontsize=100)
    ax.set_ylabel("Customer",fontsize=100)
    ax.set_xlabel("Windspeed",fontsize=100)
    plt.xticks(rotation=45)
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    st.pyplot(fig)
    
    st.caption("Windspeed has been divided to 67")

# Subheader 4
st.subheader("Bike Rental Customer Clustering by Temperature")

def scatterplot_clustering(df):
    # Standardize Features (Scaling)
    scaler = StandardScaler()
    data = df[['temp', 'cnt']]  # Select the features for clustering
    data_scaled = scaler.fit_transform(data)
    # Apply KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=40)
    df['Cluster'] = kmeans.fit_predict(data_scaled)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temp', y='cnt', hue='Cluster', palette=['purple', 'violet', 'pink'], data=df, s=100)
    plt.title("Clusters of Bike Rentals")
    plt.xlabel("Temperature")
    plt.ylabel("Count of Bike Rentals")
    plt.legend(title='Cluster', loc='upper left')
    plt.grid(True)
    st.pyplot(plt)
scatterplot_clustering(day_df)
