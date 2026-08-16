"""
Faris Alnowami
CSE 163 AC
Final Project

This module contains the comprehensive analysis functions for NHANES diabetes.
It includes demographic stratification, statistical tests with
survey-adjusted estimates, and visualization generation for undiagnosed diabetes
trends across age, race, and education levels.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import svy
from scipy.stats import chi2_contingency
from Initialization import initialization
from NHANES_Testing import validation



def NHANES_analyze(
    datasets: list[pd.DataFrame],
    full_data: pd.DataFrame,
) -> None:
    """
    Performs comprehensive NHANES diabetes epidemiology analysis on the provided
    datasets. Generates 14 visualizations showing BMI distributions, diabetes trends
    by year and demographics, age-stratified analysis, and cross-tabulations by race
    and education. Computes survey-adjusted chi-square tests using Rao-Scott
    methodology and prints statistical results and percentages for undiagnosed
    diabetes across demographic groups. Saves all figures with titles as filenames.
    """
    import seaborn as sns
    
    # BMI Distribution by Diabetes Status
    undiagnosed_bmi = full_data.loc[(full_data["Undiagnosed_Status"] == 1) & (full_data["BMXBMI"].notnull()), ["BMXBMI", "WTINT8YR"]]
    diagnosed_bmi = full_data.loc[(full_data["DIQ010"] == 1) & (full_data["BMXBMI"].notnull()), ["BMXBMI", "WTINT8YR"]]
    not_diabetic_bmi = full_data.loc[(full_data["DIQ010"] == 2) & (full_data["BMXBMI"].notnull()), ["BMXBMI", "WTINT8YR"]]
    sns.kdeplot(data=undiagnosed_bmi, x="BMXBMI", weights="WTINT8YR", label="Undiagnosed")
    sns.kdeplot(data=diagnosed_bmi, x="BMXBMI", weights="WTINT8YR", label="Diagnosed")
    sns.kdeplot(data=not_diabetic_bmi, x="BMXBMI", weights="WTINT8YR", label="Not Diabetic")
    sns.set(style="whitegrid")
    plt.xlabel("BMI")
    plt.ylabel("Density")
    plt.title("Distribution of BMI by diabetes status")
    plt.legend()
    plt.savefig("Distribution of BMI by diabetes status.png")
    plt.close()
    
    # Undiagnosed Diabetes by Year
    year_diabetes = [x.loc[x['Undiagnosed_Status'] == 1, 'WTINT2YR'].sum()/x['WTINT2YR'].sum()*100 for x in datasets]
    years = ["07-08", "09-10", "11-12", "13-14"]
    plt.bar(years, year_diabetes)
    plt.title('Percentage of People With Undiagnosed Diabetes by Year')
    plt.xlabel('Years 20XX')
    plt.ylabel('Percentage')
    plt.savefig("Percentage of People With Undiagnosed Diabetes by Year.png")
    plt.close()
    
    year_diabetes = [x.loc[x['Undiagnosed_Status'] == 1, 'WTINT2YR'].sum()/4000000 for x in datasets]
    years = ["07-08", "09-10", "11-12", "13-14"]
    plt.bar(years, year_diabetes)
    plt.title('Population Size of Undiagnosed Diabetes by Year')
    plt.xlabel('Years 20XX')
    plt.ylabel('Population Size(Millions)')
    plt.savefig("Population Size of Undiagnosed Diabetes by Year.png")
    plt.close()
    
    # Print the actual numbers for the graphs
    year_diabetes_pct = [x.loc[x['Undiagnosed_Status'] == 1, 'WTINT2YR'].sum()/x['WTINT2YR'].sum()*100 for x in datasets]
    years = ["07-08", "09-10", "11-12", "13-14"]
    
    print("\n=== Percentage of People With Undiagnosed Diabetes by Year ===")
    for year, pct in zip(years, year_diabetes_pct):
        print(f"{year}: {pct:.2f}%")
    
    print("\n=== Population Size of Undiagnosed Diabetes by Year (in Millions) ===")
    year_diabetes_pop = [x.loc[x['Undiagnosed_Status'] == 1, 'WTINT2YR'].sum()/4000000 for x in datasets]
    for year, pop in zip(years, year_diabetes_pop):
        print(f"{year}: {pop:.2f} million")
    
    # Age Trend Analysis
    age = full_data[(full_data["Undiagnosed_Status"] == 1) & (full_data["RIDAGEYR"] >= 30)].groupby("RIDAGEYR")["WTMEC8YR"].sum()/full_data[(full_data["RIDAGEYR"] >= 30)].groupby("RIDAGEYR")["WTMEC8YR"].sum()*100
    plt.figure(figsize=(11, 6))
    plt.plot(age.index, age)
    
    z = np.polyfit(age.index, age.values, 1)
    p = np.poly1d(z)
    plt.plot(age.index, p(age.index), "r--", label='Line of Best Fit')
    
    plt.title('Percentage of Undiagnosed Diabetes by Age (30+)')
    plt.xlabel('Age')
    plt.ylabel('Percentage')
    plt.legend()
    plt.savefig("Percentage of Undiagnosed Diabetes by Age (30+).png")
    plt.close()
    
    # BMI Distribution by Race
    race_labels = ['Mexican American', 'Other Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black']
    for x in range(1,5):
        race_bmi = full_data.loc[(full_data["RIDRETH1"] == x) & (full_data["BMXBMI"].notnull()), ["BMXBMI", "WTINT8YR"]]
        sns.kdeplot(data=race_bmi, x="BMXBMI", weights="WTINT8YR", label=race_labels[x-1])
    sns.set(style="whitegrid")
    plt.xlabel("BMI")
    plt.ylabel("Density")
    plt.title("Distribution of BMI by Race")
    plt.legend()
    plt.savefig("Distribution of BMI by Race.png")
    plt.close()
    
    # BMI Distribution by Education Level
    education_labels = ['Less than 9th grade', '9-11th grade', 'High school graduate', 'Some college', 'College graduate or above']
    for x in range(1,6):
        education_bmi = full_data.loc[(full_data["DMDEDUC2"] == x) & (full_data["BMXBMI"].notnull()), ["BMXBMI", "WTINT8YR"]]
        sns.kdeplot(data=education_bmi, x="BMXBMI", weights="WTINT8YR", label=education_labels[x-1])
    sns.set(style="whitegrid")
    plt.xlabel("BMI")
    plt.ylabel("Density")
    plt.title("Distribution of BMI by Education Level")
    plt.legend()
    plt.savefig("Distribution of BMI by Education Level.png")
    plt.close()
    
    education_labels = ['Less than 9th grade', '9-11th grade', 'High school graduate', 'Some college', 'College graduate or above']
    for x in range(1,6):
        education_bmi = full_data.loc[(full_data["DMDEDUC2"] == x) & (full_data["BMXBMI"].notnull()) & (full_data["RIDAGEYR"] >= 40), ["BMXBMI", "WTINT8YR"]]
        sns.kdeplot(data=education_bmi, x="BMXBMI", weights="WTINT8YR", label=education_labels[x-1])
    sns.set(style="whitegrid")
    plt.xlabel("BMI")
    plt.ylabel("Density")
    plt.title("Distribution of BMI by Education Level (Age 40+)")
    plt.legend()
    plt.savefig("Distribution of BMI by Education Level (Age 40+).png")
    plt.close()
    
    # Undiagnosed Diabetes by Race
    race_labels = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black', 5: 'Other Race'}
    plt.figure(figsize=(11, 6))
    race_undiangosed = [full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i), 'WTMEC8YR'].sum()/full_data.loc[full_data['RIDRETH1'] == i, 'WTMEC8YR'].sum()*100 for i in full_data['RIDRETH1'].unique()]
    plt.bar(full_data['RIDRETH1'].unique(), race_undiangosed)
    plt.xticks(full_data['RIDRETH1'].unique(), [race_labels[i] for i in full_data['RIDRETH1'].unique()])
    plt.title('Percentage of People With Undiagnosed Diabetes by Race')
    plt.xlabel('Race')
    plt.ylabel('Percentage')
    plt.savefig("Percentage of People With Undiagnosed Diabetes by Race.png")
    plt.close()
    
    plt.figure(figsize=(11, 6))
    race_undiangosed_count = [full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i), 'WTMEC8YR'].sum()/1000000 for i in full_data['RIDRETH1'].unique()]
    plt.bar(full_data['RIDRETH1'].unique(), race_undiangosed_count)
    plt.xticks(full_data['RIDRETH1'].unique(), [race_labels[i] for i in full_data['RIDRETH1'].unique()])
    plt.title('Number of People With Undiagnosed Diabetes by Race')
    plt.xlabel('Race')
    plt.ylabel('Number of People (Millions)')
    plt.savefig("Number of People With Undiagnosed Diabetes by Race.png")
    plt.close()
    
    print("\n=== Percentage of People With Undiagnosed Diabetes by Race ===")
    race_labels = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black', 5: 'Other Race'}
    minority_percent = 0
    for i in range(1, 6):
        pct = full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i), 'WTMEC8YR'].sum()/full_data.loc[full_data['RIDRETH1'] == i, 'WTMEC8YR'].sum()*100
        if i != 3:
            minority_percent += pct
        print(f"{race_labels[i]}: {pct:.2f}%")
    print(f"Minority Percent: {minority_percent/4:.2f}%")
    print("\n=== Number of People With Undiagnosed Diabetes by Race (in Millions) ===")
    for i in range(1, 6):
        count = full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i), 'WTMEC8YR'].sum()/1000000
        print(f"{race_labels[i]}: {count:.2f} million")
    
    # Education Level Analysis
    plt.figure(figsize=(11, 6))
    edu_data = full_data[(full_data['DMDEDUC2'].notna()) & (full_data['DMDEDUC2'] != 9) & (full_data['DMDEDUC2'] != 7)]
    education_a1c = [edu_data.loc[(edu_data['Undiagnosed_Status'] == 1) & (edu_data['DMDEDUC2'] == i) & (edu_data["RIDAGEYR"] >= 40), 'WTMEC8YR'].sum()/edu_data.loc[(edu_data['DMDEDUC2'] == i) & (edu_data["RIDAGEYR"] >= 40), 'WTMEC8YR'].sum()*100 for i in edu_data['DMDEDUC2'].unique()]
    plt.bar(edu_data['DMDEDUC2'].unique(), education_a1c)
    education_labels = {1: 'Less than 9th grade', 2: '9-11th grade', 3: 'High school graduate', 4: 'Some college', 5: 'College graduate or above'}
    plt.bar(edu_data['DMDEDUC2'].unique(), education_a1c)
    plt.xticks(edu_data['DMDEDUC2'].unique(), [education_labels[i] for i in edu_data['DMDEDUC2'].unique()])
    plt.title('Percentage of People With Undiagnosed Diabetes by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Percentage')
    plt.savefig("Percentage of People With Undiagnosed Diabetes by Education Level.png")
    plt.close()
    
    plt.figure(figsize=(11, 6))
    edu_data = full_data[(full_data['DMDEDUC2'].notna()) & (full_data['DMDEDUC2'] != 9) & (full_data['DMDEDUC2'] != 7)]
    education_a1c = [edu_data.loc[(edu_data['Undiagnosed_Status'] == 1) & (edu_data['DMDEDUC2'] == i) & (edu_data["RIDAGEYR"] >= 40), 'WTMEC8YR'].sum()/1000000 for i in edu_data['DMDEDUC2'].unique()]
    plt.bar(edu_data['DMDEDUC2'].unique(), education_a1c)
    plt.bar(edu_data['DMDEDUC2'].unique(), education_a1c)
    plt.xticks(edu_data['DMDEDUC2'].unique(), [education_labels[i] for i in edu_data['DMDEDUC2'].unique()])
    plt.title('Number of People With Undiagnosed Diabetes by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Number of People(Millions)')
    plt.savefig("Number of People With Undiagnosed Diabetes by Education Level.png")
    plt.close()
    
    print("\n=== Percentage of People With Undiagnosed Diabetes by Education Level (Age 40+) ===")
    edu_data = full_data[(full_data['DMDEDUC2'].notna()) & (full_data['DMDEDUC2'] != 9) & (full_data['DMDEDUC2'] != 7)]
    education_labels = {1: 'Less than 9th grade', 2: '9-11th grade', 3: 'High school graduate', 4: 'Some college', 5: 'College graduate or above'}
    for i in range(1, 6):
        pct = edu_data.loc[(edu_data['Undiagnosed_Status'] == 1) & (edu_data['DMDEDUC2'] == i) & (edu_data["RIDAGEYR"] >= 40), 'WTMEC8YR'].sum()/edu_data.loc[(edu_data['DMDEDUC2'] == i) & (edu_data["RIDAGEYR"] >= 40), 'WTMEC8YR'].sum()*100
        print(f"{education_labels[i]}: {pct:.2f}%")
    
    print("\n=== Number of People With Undiagnosed Diabetes by Education Level (Age 40+) (in Millions) ===")
    for i in range(1, 6):
        count = edu_data.loc[(edu_data['Undiagnosed_Status'] == 1) & (edu_data['DMDEDUC2'] == i) & (edu_data["RIDAGEYR"] >= 40), 'WTMEC8YR'].sum()/1000000
        print(f"{education_labels[i]}: {count:.2f} million")
    
    # Chi-Square Test: Race and Undiagnosed Status
    table = pd.crosstab(
        full_data["RIDRETH1"],
        full_data["Undiagnosed_Status"]
    )
    
    print("Unweighted contingency table:")
    print(table)
    
    _, _, _, expected = chi2_contingency(table)
    
    print("\nExpected counts:")
    print(expected)
    print("\nMinimum expected count:", expected.min())
    
    svy_data = pl.from_pandas(full_data[full_data["Undiagnosed_Status"].notna()])
    
    design = svy.Design(
        stratum="SDMVSTRA",
        psu="SDMVPSU",
        wgt="WTMEC8YR"
    )
    
    sample = svy.Sample(
        data=svy_data,
        design=design
    )
    
    result = sample.categorical.tabulate(
        rowvar="RIDRETH1",
        colvar="Undiagnosed_Status",
        units="percent"
    )
    
    print("\nSurvey-adjusted results:")
    print(result)

    print("Chi-Squared Test Result of Race and Undiagnosed Diabetes")
    print("First-order Rao-Scott:")
    print("Chi-square:", result.stats.chisq.value)
    print("df:", result.stats.chisq.df)
    print("p-value:", result.stats.chisq.p_value)
    
    print("\nSecond-order Rao-Scott:")
    print("F:", result.stats.f.value)
    print("Numerator df:", result.stats.f.df_num)
    print("Denominator df:", result.stats.f.df_den)
    print("p-value:", result.stats.f.p_value)
    
    # Chi-Square Test: Education and Undiagnosed Status
    table = pd.crosstab(
        full_data["DMDEDUC2"],
        full_data["Undiagnosed_Status"]
    )
    
    print("Unweighted contingency table:")
    print(table)
    
    _, _, _, expected = chi2_contingency(table)
    
    print("\nExpected counts:")
    print(expected)
    print("\nMinimum expected count:", expected.min())
    
    svy_data = pl.from_pandas(full_data[full_data["Undiagnosed_Status"].notna() & full_data["DMDEDUC2"].notna()])
    
    design = svy.Design(
        stratum="SDMVSTRA",
        psu="SDMVPSU",
        wgt="WTMEC8YR"
    )
    
    sample = svy.Sample(
        data=svy_data,
        design=design
    )
    
    result = sample.categorical.tabulate(
        rowvar="DMDEDUC2",
        colvar="Undiagnosed_Status",
        units="percent"
    )
    
    print("\nSurvey-adjusted results:")
    print(result)

    print("Chi-Squared Test Result of Education and Undiagnosed Diabetes")
    print("First-order Rao-Scott:")
    print("Chi-square:", result.stats.chisq.value)
    print("df:", result.stats.chisq.df)
    print("p-value:", result.stats.chisq.p_value)
    
    print("\nSecond-order Rao-Scott:")
    print("F:", result.stats.f.value)
    print("Numerator df:", result.stats.f.df_num)
    print("Denominator df:", result.stats.f.df_den)
    print("p-value:", result.stats.f.p_value)
    
    # Race and Education Cross-Tabulation
    race_labels = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black'}
    plt.figure(figsize=(11, 6))
    race_undiangosed = [full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 5), 'WTMEC8YR'].sum()/full_data.loc[(full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 5), 'WTMEC8YR'].sum()*100 for i in range(1,5)]
    plt.bar(range(1,5), race_undiangosed)
    plt.xticks(range(1,5), [race_labels[i] for i in range(1,5)])
    plt.title('Percentage of College Educated People With Undiagnosed Diabetes by Race')
    plt.xlabel('Race')
    plt.ylabel('Percentage')
    plt.savefig("Percentage of College Educated People With Undiagnosed Diabetes by Race.png")
    plt.close()
    
    race_labels = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black'}
    plt.figure(figsize=(11, 6))
    race_undiangosed = [full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 3), 'WTMEC8YR'].sum()/full_data.loc[(full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 3), 'WTMEC8YR'].sum()*100 for i in range(1,5)]
    plt.bar(range(1,5), race_undiangosed)
    plt.xticks(range(1,5), [race_labels[i] for i in range(1,5)])
    plt.title('Percentage of High School Educated People With Undiagnosed Diabetes by Race')
    plt.xlabel('Race')
    plt.ylabel('Percentage')
    plt.savefig("Percentage of High School Educated People With Undiagnosed Diabetes by Race.png")
    plt.close()
    
    race_labels = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black', 5: 'Other Race'}
    plt.figure(figsize=(11, 6))
    race_undiangosed = [full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 1), 'WTMEC8YR'].sum()/full_data.loc[(full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 1), 'WTMEC8YR'].sum()*100 for i in full_data['RIDRETH1'].unique()]
    plt.bar(full_data['RIDRETH1'].unique(), race_undiangosed)
    plt.xticks(full_data['RIDRETH1'].unique(), [race_labels[i] for i in full_data['RIDRETH1'].unique()])
    plt.title('Percentage of Less than 9th Grade Educated People With Undiagnosed Diabetes by Race')
    plt.xlabel('Race')
    plt.ylabel('Percentage')
    plt.savefig("Percentage of Less than 9th Grade Educated People With Undiagnosed Diabetes by Race.png")
    plt.close()
    
    print("\n=== College Educated People With Undiagnosed Diabetes by Race ===")
    race_labels = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black', 5: 'Other Race'}
    for i in range(1, 5):
        pct = full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 5), 'WTMEC8YR'].sum()/full_data.loc[(full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 5), 'WTMEC8YR'].sum()*100
        print(f"{race_labels[i]}: {pct:.2f}%")
    
    print("\n=== High School Educated People With Undiagnosed Diabetes by Race ===")
    for i in range(1, 5):
        pct = full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 3), 'WTMEC8YR'].sum()/full_data.loc[(full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 3), 'WTMEC8YR'].sum()*100
        print(f"{race_labels[i]}: {pct:.2f}%")
    
    print("\n=== Less than 9th Grade Educated People With Undiagnosed Diabetes by Race ===")
    for i in range(1, 5):
        pct = full_data.loc[(full_data['Undiagnosed_Status'] == 1) & (full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 1), 'WTMEC8YR'].sum()/full_data.loc[(full_data['RIDRETH1'] == i) & (full_data["DMDEDUC2"] == 1), 'WTMEC8YR'].sum()*100
        print(f"{race_labels[i]}: {pct:.2f}%")
    


if __name__ == "__main__":
    datasets, full_data = initialization()
    validation(datasets, full_data)
    
    NHANES_analyze(datasets, full_data)
    
