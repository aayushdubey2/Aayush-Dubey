import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
covid=pd.read_csv("covid_19.csv")
print(covid.isnull().sum())
print(covid.dtypes)
covid.drop(["Province/State"],axis=1,inplace=True)
covid["ObservationDate"]=pd.to_datetime(covid["ObservationDate"])
print(covid.head(20))
#%%
#Country-wise data
grouped_country=covid.groupby(["Country/Region","ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
grouped_country["Active Cases"]=grouped_country["Confirmed"]-grouped_country["Recovered"]-grouped_country["Deaths"]
grouped_country["log_confirmed"]=np.log(grouped_country["Confirmed"])
grouped_country["log_active"]=np.log(grouped_country["Active Cases"])
print(grouped_country.head())
#%%
# Datewise Analysis
datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":"sum","Recovered":"sum","Deaths":"sum"})
datewise["Days Since"]=datewise.index-datewise.index.min()
print(datewise.head())
#%%
#BASIC INFORMATION
print("Total no of countries/Regions with corona cases: ", len(covid["Country/Region"].unique()))
print("Total Number of Confirmed cases around the world: ",datewise["Confirmed"][-1])
print("Total Number of Recovered cases around the world: ",datewise["Recovered"][-1])
print("Total Number of Death cases around the world: ",datewise["Deaths"][-1])
print("Total number of active cases in the world: ",datewise["Confirmed"][-1]-datewise["Recovered"][-1]-datewise["Deaths"][-1])
print("Avg number of confirmed cases per day: ",int(int(datewise["Confirmed"][-1])/len(datewise)))
print("Avg number of recovered cases per day: ",int(int(datewise["Recovered"][-1])/len(datewise)))
print("Avg number of deaths per day: ",int(int(datewise["Deaths"][-1])/len(datewise)))
print("Approximate number of confirmed cases per hour: ",int(int(datewise["Confirmed"][-1])/(len(datewise)*24)))
print("Approximate number of Recovered cases per hour: ",int(int(datewise["Recovered"][-1])/(len(datewise)*24)))
print("Approximate number of Deaths per hour: ",int(int(datewise["Deaths"][-1])/(len(datewise)*24)))
print("Total no of recovered cases in last 1 day:",datewise["Recovered"][-1]-datewise["Recovered"][-2])
print("Total no of confirmed cases in last 1 day:",datewise["Confirmed"][-1]-datewise["Confirmed"][-2])
print("Total no of deaths in last 1 day:",datewise["Deaths"][-1]-datewise["Deaths"][-2])
#%%
#Over-allanalysis

plt.figure(figsize=(25,5))
sns.barplot(x=datewise.index.date,y=datewise["Confirmed"])
plt.title("Plot for confirmed cases over date")
plt.xlabel("Date")
plt.ylabel("Number of confirmed cases")
plt.xticks(rotation=90)


plt.figure(figsize=(25,5))
sns.barplot(x=datewise.index.date,y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])
plt.title("Plot for active cases over date")
plt.xlabel("Date")
plt.ylabel("Number of active cases")
plt.xticks(rotation=90)

plt.figure(figsize=(25,5))
sns.barplot(x=datewise.index.date,y=datewise["Recovered"])
plt.title("Plot for recovered cases over date")
plt.xlabel("Date")
plt.ylabel("Number of recoveries")
plt.xticks(rotation=90)

plt.figure(figsize=(25,5))
sns.barplot(x=datewise.index.date,y=datewise["Deaths"])
plt.title("Plot for deaths over date")
plt.xlabel("Date")
plt.ylabel("Number of deaths")
plt.xticks(rotation=90)

#%%
# Weekwise Analysis
datewise["weekofyear"]=datewise.index.week
print(datewise.head())
confirmed_weekwise=[]
recovered_weekwise=[]
death_weekwise=[]

yy=list(set(datewise["weekofyear"]))
for i in yy:
    confirmed_weekwise.append(datewise[datewise["weekofyear"]==i]["Confirmed"][-1])
    recovered_weekwise.append(datewise[datewise["weekofyear"]==i]["Recovered"][-1])
    death_weekwise.append(datewise[datewise["weekofyear"]==i]["Deaths"][-1])
     
#%%
# Cases over week
plt.figure(figsize=(10,7))
ax=sns.lineplot(x=yy,y=confirmed_weekwise, color="blue",label="Confirmed",marker="o")
ax=sns.lineplot(x=yy, y=recovered_weekwise,color="green", label="Recovered",marker="o")
ax=sns.lineplot(x=yy,y=death_weekwise,color="red",label="Deaths",marker="o")
ax.legend()
plt.title("Cases over week")
#%%
# diff is used to subtract the previous value from current value, and here deleting prev value fron curr give us weekly cases.
#https://www.w3resource.com/pandas/dataframe/dataframe-diff.php#:~:text=The%20diff()%20function%20calculates,another%20element%20in%20the%20DataFrame.&text=Periods%20to%20shift%20for%20calculating%20difference%2C%20accepts%20negative%20values.&text=Take%20difference%20over%20rows%20(0)%20or%20columns%20(1).
#cases per week
plt.figure(figsize=(10,5))
sns.barplot(x=yy,y=pd.Series(confirmed_weekwise).diff().fillna(confirmed_weekwise[0]))
plt.title("No of confirmed cases per week")
plt.xlabel("Week of the year")
plt.ylabel("Number of Confirmed cases")

plt.figure(figsize=(10,5))
sns.barplot(x=yy,y=pd.Series(recovered_weekwise).diff().fillna(recovered_weekwise[0]))
plt.title("No of recovered cases per week")
plt.xlabel("Week of the year")
plt.ylabel("Number of recovered cases")

plt.figure(figsize=(10,5))
sns.barplot(x=yy,y=pd.Series(death_weekwise).diff().fillna(death_weekwise[0]))
plt.title("No of deaths per week")
plt.xlabel("Week of the year")
plt.ylabel("Number of deaths")

#%%
# Recovery Rate and Mortality rate
datewise["recovery rate"]=(datewise["Recovered"]/datewise["Confirmed"])*100
datewise["mortality rate"]=(datewise["Deaths"]/datewise["Confirmed"])*100
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
sns.lineplot(x=datewise.index,y=datewise["recovery rate"],data=datewise, color="green")
plt.text(datewise.index[60],datewise["recovery rate"][-2],"avg recovery rate="+str(round(datewise["recovery rate"].mean(),2)))
plt.figure(figsize=(10,7))
plt.subplot(2,1,2)
sns.lineplot(x=datewise.index,y=datewise["mortality rate"],data=datewise, color="red")
plt.text(datewise.index[0],datewise["mortality rate"][-2],"avg mortality rate="+str(round(datewise["mortality rate"].mean(),2)))
#%%
#Growth Factor
plt.figure(figsize=(10,7))
print("Avg growth factor of no of confirmed cases:",(datewise["Confirmed"]/datewise["Confirmed"].shift()).mean())
print("Avg growth factor of no of recovered cases:",(datewise["Recovered"]/datewise["Recovered"].shift()).mean())
print("Avg growth factor of no of death cases:",(datewise["Deaths"]/datewise["Deaths"].shift()).mean())
sns.lineplot(x=datewise.index,y=(datewise["Confirmed"]/datewise["Confirmed"].shift()),color="blue",label="confirmed")
sns.lineplot(x=datewise.index,y=(datewise["Recovered"]/datewise["Recovered"].shift()),color="green",label="recovered")
sns.lineplot(x=datewise.index,y=(datewise["Deaths"]/datewise["Deaths"].shift()),color="red",label="deaths")
plt.ylabel("Avg growth rate")
#%%

# Days taken in doubling of cases
c=555
double_days=[]
C=[]
while(1):
    double_days.append(datewise[datewise["Confirmed"]<=c].iloc[[-1]]["Days Since"][0])
    C.append(c)
    c=c*2
    if(c<datewise["Confirmed"].max()):
        continue
    else:
        break
doubling_rate=pd.DataFrame(list(zip(C,double_days)),columns=["No. of cases","Days since first Case"])
doubling_rate["Number of days for doubling"]=doubling_rate["Days since first Case"].diff().fillna(doubling_rate["Days since first Case"])
doubling_rate
#%%
# Top 15 countries
countrywise=covid[covid["ObservationDate"]==covid["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed","Recovered","Deaths"],ascending=False)

fig, (ax1, ax2,ax3) = plt.subplots(3, 1,figsize=(10,18))
top_15_confirmed=countrywise.sort_values(["Confirmed"],ascending=False).head(15)
top_15_recovered=countrywise.sort_values(["Recovered"],ascending=False).head(15)
top_15_deaths=countrywise.sort_values(["Deaths"],ascending=False).head(15)
sns.barplot(x=top_15_confirmed["Confirmed"],y=top_15_confirmed.index,ax=ax1)
ax1.set_title("Top 15 countries as per Number of Confirmed Cases")
sns.barplot(x=top_15_recovered["Recovered"],y=top_15_recovered.index,ax=ax2)
ax2.set_title("Top 15 countries as per Number of Recovered Cases")
sns.barplot(x=top_15_deaths["Deaths"],y=top_15_deaths.index,ax=ax3)
ax3.set_title("Top 15 countries as per Number of Death Cases")
