import pandas as pd
import numpy as np
import os, sys


def per_million(data, population):
    # divide by population
    data_per_mio_capita = np.zeros_like(data)
    for i in range(len(population)):
        data_per_mio_capita[i, :] = data[i, :] / population[i] * 1e6
    return data_per_mio_capita


def get_daily(data):
    # get daily cases by taking 1st order differencing
    data = np.diff(data)
    data[data < 0] = 0
    # # drop first day because daily_case arrays has 1 less data on time dimension
    # confirmed_cases = confirmed_cases[:,1:]
    # confirmed_filtered = confirmed_filtered[:,1:]
    return data


def load_data(DATASET, path="/content/drive/Shareddrives/covid.eng.pdn.ac.lk/COVID-AI (PG)/spatio_temporal/Datasets"):
    if DATASET == "Sri Lanka":
        dataset_path = os.path.join(path, "SL")

        df_confirmed = pd.read_csv(os.path.join(dataset_path, "SL_covid_all_updated.csv"))
        df_confirmed = df_confirmed.set_index("Code").sort_index()
        df_population = pd.read_csv(os.path.join(dataset_path, "SL_population_updated.csv"))
        df_population = df_population.set_index("Code").sort_index()

        df_food_ratios = pd.read_csv(os.path.join(dataset_path, "foodexpenditure_ratios_updated.csv"))
        df_food_ratios = df_food_ratios.set_index("Code").sort_index()

        df_unemployment = pd.read_csv(os.path.join(dataset_path, "unemployment_updated.csv"))
        df_unemployment = df_unemployment.set_index("Code").sort_index()

        df_poverty = pd.read_csv(os.path.join(dataset_path, "povery_rates2012_updated.csv"))
        df_poverty = df_poverty.set_index("Code").sort_index()

        df_internet = pd.read_csv(os.path.join(dataset_path, "internet_percent_women_updated.csv"))
        df_internet = df_internet.set_index("Code").sort_index()

        df_education = pd.read_csv(os.path.join(dataset_path, "education_years_women_updated.csv"))
        df_education = df_education.set_index("Code").sort_index()

        df_industry = pd.read_csv(os.path.join(dataset_path, "industry_women_updated.csv"))
        df_industry = df_industry.set_index("Code").sort_index()

        # Dropping Kalmunai
        df_confirmed = df_confirmed.loc[df_confirmed['District'] != 'KALMUNAI']
        df_confirmed = df_confirmed.rename(columns={"District": "Region"})

        region_codes = df_confirmed.index

        confirmed_cases = np.array(np.float64(df_confirmed.iloc[0:25, 1:].values))
        daily_cases = np.diff(confirmed_cases)

        region_names = list(df_confirmed['Region'])
        region_names[25:] = []

        population = 1000 * df_population.iloc[:, 7]
        lat = df_population["Lat"]
        lon = df_population["Lon"]
        land = df_population["Land Area"]
        pop_density = pd.Series(population / land, name="Population density")
        labour_total = df_industry["Total percentage of persons involved in labour"]
        labour_skilled = df_industry["Percentage of persons involved in skilled labour"]
        labour_unskilled = df_industry["Percentage of persons involved in unskilled labour"]
        labour_agri = df_industry["Percentage of persons involved in agriculture"]

        unemployment = df_unemployment["Unemployment rate"]
        poverty = df_poverty["Poverty rate"]

        spending_total = df_food_ratios["Total monthly expenditure"]
        spending_food = df_food_ratios["Monthly expenditure on food and drink"]
        spending_other = df_food_ratios["Monthly expenditure on non-food items"]
        spending_ratio = df_food_ratios["Ratio between expenditure on food and non-food items"]

        internet = df_internet["Percentage of persons using internet"]
        education = df_education["Median years spent in education"]

        features = pd.concat(
            [population, lat, lon, pop_density, labour_total, labour_skilled, labour_unskilled, labour_agri,
             unemployment, poverty, spending_total, spending_food, spending_other, spending_ratio, internet, education],
            axis=1, join="inner").rename(
            columns={"Total (2017)": "Population",
                     "Total percentage of persons involved in labour": "Labour(Total %)",
                     "Percentage of persons involved in skilled labour": "Labour(Skilled %)",
                     "Percentage of persons involved in unskilled labour": "Labour(Unskilled %)",
                     "Percentage of persons involved in agriculture": "Labour(Agri %)",
                     "Total monthly expenditure": "Expenses(Total)",
                     "Monthly expenditure on food and drink": "Expenses(Food)",
                     "Monthly expenditure on non-food items": "Expenses(Non-food)",
                     "Ratio between expenditure on food and non-food items": "Expenses(Food/Non-food)",
                     "Percentage of persons using internet": "Internet usage(%)",
                     "Median years spent in education": "Education(Median years)"})

        n_regions = confirmed_cases.shape[0]

        START_DATE = "14/11/2020"
        print(confirmed_cases.shape)
        print(region_names)
        print(region_codes)

    if DATASET == "Texas":
        START_DATE = "03/04/2020"

        dataset_path = os.path.join(path, "Texas")

        # dataframes
        df_confirmed = pd.read_csv(os.path.join(dataset_path, "Texas COVID-19 Case Count Data by County.csv"),
                                   skiprows=2, nrows=254)  # https://dshs.texas.gov/coronavirus/AdditionalData.aspx
        df_population = pd.read_csv(os.path.join(dataset_path, "2019_txpopest_county.csv"),
                                    header=0)  # https://demographics.texas.gov/Resources/TPEPP/Estimates/2019/2019_txpopest_county.csv

        print(df_population)

        # conv to np.array
        confirmed_cases = np.array(np.float64(df_confirmed.iloc[:, 1:].values))
        region_names = np.array(df_population.iloc[:-1, 1].values)

        population = df_population.iloc[:-1, 2]
        features = pd.concat([population], axis=1, join="inner").rename(columns={'cqr_census_2010_count': 'Population'})

        # fixing the confirmed cases dataset (negative gradients)
        for k in range(confirmed_cases.shape[0]):
            for i in range(confirmed_cases.shape[1] - 1):
                if confirmed_cases[k, i + 1] < confirmed_cases[k, i]:
                    confirmed_cases[k, i + 1] = confirmed_cases[k, i]

        daily_cases = np.diff(confirmed_cases, axis=-1)

        n_regions = confirmed_cases.shape[0]
        days = confirmed_cases.shape[1]

        print('confirmed cases shape:', confirmed_cases.shape, '  daily cases shape:', daily_cases.shape,
              '  population shape:', population.shape)
        print('counties:', n_regions, '  days:', days)

        # daily_cases_1M = np.copy(daily_cases)
        # confirmed_cases_1M = np.copy(confirmed_cases)
        # for i in range(n_regions):
        #     daily_cases_1M[i,:] = 1000000*daily_cases_1M[i,:]/population[i]
        #     confirmed_cases_1M[i,:] = 1000000*confirmed_cases_1M[i,:]/population[i]

        # plots = [confirmed_cases[:,:].T, daily_cases[:,:].T, confirmed_cases_1M[:,:].T, daily_cases_1M[:,:].T, ]
        # titles = ['Texas: cumulative cases','Texas: daily new cases','Texas: cumulative cases per 1M','Texas: daily new cases per 1M']
        # plt.figure(figsize=(14,9))
        # for i in range(len(titles)):
        #     plt.subplot(2,2,i+1)
        #     plt.plot(plots[i],linewidth=2)
        #     plt.title(titles[i])
        #     if i in [2,3]:
        #         plt.xlabel('days since April 3 2020')
        # plt.show()
    if DATASET == "USA":
        dataset_path = os.path.join(path, "US")

        state_names = pd.read_csv(os.path.join(dataset_path, "state_name.csv"), header=None)
        state_names.columns = ['State Name', 'State Code']

        df_daily = pd.read_csv(os.path.join(dataset_path, "cases_new.csv"), header=None)

        region_names = [state_names.iloc[i, 0] for i in range(N)]
        daily_cases = np.array(np.float64(df_daily.iloc[:, :].values))
        confirmed_cases = np.cumsum(daily_cases, axis=1)

        # features
        health = pd.read_csv(os.path.join(dataset_path, "healthins.csv"), header=None)
        povert = pd.read_csv(os.path.join(dataset_path, "poverty.csv"), header=None)
        income = pd.read_csv(os.path.join(dataset_path, "income.csv"), header=None)
        popden = pd.read_csv(os.path.join(dataset_path, "pop_density.csv"), header=None)
        population = pd.read_csv(os.path.join(dataset_path, "pop.csv"), header=None)
        features = pd.concat([population, popden, health, income, povert], axis=1)

        START_DATE = "14/01/2020"  # TODO FIND

    if DATASET == "NG":
        dataset_path = os.path.join(path, "NG")
        df_daily = pd.read_excel(os.path.join(dataset_path, "nga_subnational_covid19_hera.xls"))
        df_daily = df_daily[['DATE','REGION','CONTAMINES']]
        dates = df_daily['DATE'].unique()
        region_names = df_daily['REGION'].unique()
        daily_cases = pd.DataFrame(columns=dates, index=region_names)

        for date in dates:
        	df_date = df_daily.loc[df_daily['DATE']==date]
        	df_date = df_date[['REGION','CONTAMINES']]
        	df_date = df_date.set_index('REGION')
        	daily_cases.loc[df_date.index, date] = df_date.values.reshape(-1) 
        
        daily_cases = np.array(np.float64(daily_cases.values))
        confirmed_cases = np.cumsum(daily_cases, axis=1)
        np.save('ng',daily_cases)
        START_DATE = '2/27/2020'
        n_regions = len(region_names)
    if DATASET == "Global":
        dataset_path = os.path.join(path, "Global")
    return {
        "region_names": region_names,
        "confirmed_cases": confirmed_cases,
        "daily_cases": daily_cases,
        "features": features,
        "START_DATE": START_DATE,
        "n_regions": n_regions,
    }

if __name__ == "__main__":
    load_data("NG","../../Datasets")