"""
Faris Alnowami
CSE 163 AC
Final Project

This module contains the initialization and summary functions for the NHANES
dataset analysis. It includes functions to load and merge the datasets,
and generate summary statistics and plots for the project-related variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW


def initialization() -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """
    Loads and merges the NHANES datasets into yearly datasets and a combined
    dataset. It also calculates undiagnosed status, summary statistics, and
    missing-value percentages for each dataset and the combined dataset.
    Returns a list of yearly datasets and the combined dataset.
    """
    base_path = "C:\\Users\\faris\\Downloads\\Research-Data\\Research-Data"

    DIQ_E = pd.read_sas(f"{base_path}\\DIQ_E.xpt")
    DIQ_F = pd.read_sas(f"{base_path}\\DIQ_F.xpt")
    DIQ_G = pd.read_sas(f"{base_path}\\DIQ_G.xpt")
    DIQ_H = pd.read_sas(f"{base_path}\\DIQ_H.xpt")

    DEMO_E = pd.read_sas(f"{base_path}\\DEMO_E.xpt")
    DEMO_F = pd.read_sas(f"{base_path}\\DEMO_F.xpt")
    DEMO_G = pd.read_sas(f"{base_path}\\DEMO_G.xpt")
    DEMO_H = pd.read_sas(f"{base_path}\\DEMO_H.xpt")

    GHB_E = pd.read_sas(f"{base_path}\\GHB_E.xpt")
    GHB_F = pd.read_sas(f"{base_path}\\GHB_F.xpt")
    GHB_G = pd.read_sas(f"{base_path}\\GHB_G.xpt")
    GHB_H = pd.read_sas(f"{base_path}\\GHB_H.xpt")

    BMX_E = pd.read_sas(f"{base_path}\\BMX_E.xpt")
    BMX_F = pd.read_sas(f"{base_path}\\BMX_F.xpt")
    BMX_G = pd.read_sas(f"{base_path}\\BMX_G.xpt")
    BMX_H = pd.read_sas(f"{base_path}\\BMX_H.xpt")

    data_relation = [
        (DIQ_E, DEMO_E, GHB_E, BMX_E),
        (DIQ_F, DEMO_F, GHB_F, BMX_F),
        (DIQ_G, DEMO_G, GHB_G, BMX_G),
        (DIQ_H, DEMO_H, GHB_H, BMX_H),
    ]

    merged_years = []
    for diq, demo, ghb, bmx in data_relation:
        merged = pd.merge(diq, demo, on="SEQN", how="inner")
        merged = pd.merge(merged, ghb, on="SEQN", how="inner")
        merged = pd.merge(merged, bmx, on="SEQN", how="inner")
        merged_years.append(merged)

    data_E, data_F, data_G, data_H = merged_years

    datasets = [data_E, data_F, data_G, data_H]
    years = ["07-08", "09-10", "11-12", "13-14"]
    for i in range(len(datasets)):
        print(
            "\nPercentage of missing values in the overall",
            years[i],
            "data set:",
        )
        print(
            "# of columns: ",
            len(datasets[i].columns),
            "\n# of rows: ",
            len(datasets[i]),
            "\n",
        )
        for column in datasets[i].columns:
            percent_missing = (
                datasets[i][column].isnull().sum() / len(datasets[i]) * 100
            )
            print(column, " ", percent_missing, "%")

    for i in range(len(datasets)):
        if i < 4:
            datasets[i] = datasets[i][[
                "SEQN",
                "SDDSRVYR",
                "DIQ010",
                "RIAGENDR",
                "RIDAGEYR",
                "RIDRETH1",
                "DMDEDUC2",
                "WTINT2YR",
                "WTMEC2YR",
                "LBXGH",
                "BMXBMI",
                "SDMVSTRA",
                "SDMVPSU",
            ]]
        else:
            datasets[i] = datasets[i][[
                "SEQN",
                "SDDSRVYR",
                "DIQ010",
                "RIAGENDR",
                "RIDAGEYR",
                "RIDRETH1",
                "RIDRETH3",
                "DMDEDUC2",
                "WTINT2YR",
                "WTMEC2YR",
                "LBXGH",
            ]]
        datasets[i]["A1C_Based_Diabetes"] = np.where(
            datasets[i]["LBXGH"] >= 6.5,
            1,
            np.nan,
        )
        datasets[i]["A1C_Based_Diabetes"] = np.where(
            datasets[i]["LBXGH"] < 6.5,
            2,
            datasets[i]["A1C_Based_Diabetes"],
        )
        datasets[i]["Undiagnosed_Status"] = np.where(
            (datasets[i]["DIQ010"] == 2)
            & (datasets[i]["A1C_Based_Diabetes"] == 1),
            1,
            np.nan,
        )
        datasets[i]["Undiagnosed_Status"] = np.where(
            (datasets[i]["A1C_Based_Diabetes"] == 2),
            2,
            datasets[i]["Undiagnosed_Status"],
        )
        datasets[i]["Undiagnosed_Status"] = np.where(
            (datasets[i]["DIQ010"] != 2)
            & (datasets[i]["A1C_Based_Diabetes"] == 1),
            2,
            datasets[i]["Undiagnosed_Status"],
        )
    data_E, data_F, data_G, data_H = datasets

    years = ["07-08", "09-10", "11-12", "13-14"]
    for i in range(len(datasets)):
        print(
            "\nPercentage of missing values in the selected",
            years[i],
            "data set:",
        )
        print(
            "# of columns: ",
            len(datasets[i].columns),
            "\n# of rows: ",
            len(datasets[i]),
            "\n",
        )
        for column in datasets[i].columns:
            percent_missing = (
                datasets[i][column].isnull().sum() / len(datasets[i]) * 100
            )
            print(column, " ", percent_missing, "%")

    for i in range(len(datasets)):
        print(
            "\n-----------Description of",
            years[i],
            "data set:-----------\n",
        )
        for column in datasets[i].columns:
            if column in ["WTINT2YR", "WTMEC2YR"]:
                print(
                    "total weight of ",
                    column,
                    " is: ",
                    datasets[i][column].sum(),
                )
            elif column == "RIDAGEYR":
                print("\nSummary of RIDAGEYR: ")
                stats = DescrStatsW(
                    datasets[i][column],
                    weights=datasets[i]["WTINT2YR"],
                    ddof=0,
                )
                print(
                    "Mean:",
                    stats.mean,
                    " Std Dev:",
                    stats.std,
                    " Min:",
                    datasets[i][column].min(),
                    " Max:",
                    datasets[i][column].max(),
                    " 25th Percentile:",
                    stats.quantile(0.25).values[0],
                    " Median:",
                    stats.quantile(0.5).values[0],
                    " 75th Percentile:",
                    stats.quantile(0.75).values[0],
                )
            elif column == "LBXGH":
                print("\nSummary of LBXGH:")
                data = datasets[i][datasets[i][column].notnull()]
                stats = DescrStatsW(
                    data[column],
                    weights=data["WTMEC2YR"],
                    ddof=0,
                )
                print(
                    "Mean:",
                    stats.mean,
                    " Std Dev:",
                    stats.std,
                    " Min:",
                    data[column].min(),
                    " Max:",
                    data[column].max(),
                    " 25th Percentile:",
                    stats.quantile(0.25).values[0],
                    " Median:",
                    stats.quantile(0.5).values[0],
                    " 75th Percentile:",
                    stats.quantile(0.75).values[0],
                )
            elif column == "SEQN":
                print(
                    "\n Number of Unique SQN: ",
                    datasets[i][column].nunique(),
                )
            else:
                print("\n")
                print(
                    "Total unwighted counts for ",
                    column,
                    ":",
                    datasets[i][column].value_counts(dropna=False),
                )
                for value in datasets[i][column].unique():
                    print(
                        column,
                        " ",
                        value,
                        " count: ",
                        datasets[i].loc[
                            datasets[i][column] == value,
                            "WTINT2YR",
                        ].sum(),
                    )

    full_data = pd.concat(datasets, ignore_index=True)
    full_data["WTINT8YR"] = full_data["WTINT2YR"] / 4
    full_data["WTMEC8YR"] = full_data["WTMEC2YR"] / 4
    print("\nPercentage of missing values in the total selected data set:")
    print(
        "# of columns: ",
        len(full_data.columns),
        "\n# of rows: ",
        len(full_data),
        "\n",
    )
    for column in full_data.columns:
        percent_missing = (
            full_data[column].isnull().sum() / len(full_data) * 100
        )
        print(column, " ", percent_missing, "%")

    print("\n-----------Description of total data set:-----------\n")
    for column in full_data.columns:
        if column in ["WTINT2YR", "WTMEC2YR", "WTINT8YR", "WTMEC8YR"]:
            print("total weight of ", column, " is: ", full_data[column].sum())
        elif column == "RIDAGEYR":
            stats = DescrStatsW(
                full_data[column],
                weights=full_data["WTINT2YR"],
                ddof=0,
            )
            print("\nSummary of RIDAGEYR:")
            print(
                "Mean:",
                stats.mean,
                " Std Dev:",
                stats.std,
                " Min:",
                full_data[column].min(),
                " Max:",
                full_data[column].max(),
                " 25th Percentile:",
                stats.quantile(0.25).values[0],
                " Median:",
                stats.quantile(0.5).values[0],
                " 75th Percentile:",
                stats.quantile(0.75).values[0],
            )
        elif column == "LBXGH":
            print("\nSummary of LBXGH:")
            data = full_data[full_data[column].notnull()]
            stats = DescrStatsW(data[column], weights=data["WTMEC2YR"], ddof=0)
            print(
                "Mean:",
                stats.mean,
                " Std Dev:",
                stats.std,
                " Min:",
                full_data[column].min(),
                " Max:",
                full_data[column].max(),
                " 25th Percentile:",
                stats.quantile(0.25).values[0],
                " Median:",
                stats.quantile(0.5).values[0],
                " 75th Percentile:",
                stats.quantile(0.75).values[0],
            )
        elif column == "SEQN":
            print("\n Number of Unique SQN: ", full_data[column].nunique())
        else:
            print("\n")
            print(
                "Total unwighted counts for ",
                column,
                ":",
                full_data[column].value_counts(dropna=False),
            )
            for value in full_data[column].unique():
                print(
                    column,
                    " ",
                    value,
                    " count: ",
                    full_data.loc[
                        full_data[column] == value,
                        "WTINT2YR",
                    ].sum(),
                )
    return datasets, full_data


def plot_summary_graphs(
    datasets: list[pd.DataFrame],
    full_data: pd.DataFrame,
) -> None:
    """
    Given a list of the yearly datasets and the combined dataset,
    creates and displays diabetes-related summary graphs, including the
    percentage of people with diabetes by year, the percentage of people
    with diabetic level A1C by year, and the percentage of people with
    diabetic level A1C by race and education level.
    """
    years = ["07-08", "09-10", "11-12", "13-14"]
    output_dir = "C:\\Users\\faris\\Downloads\\Research-Data\\Research-Data"

    year_diabetes = [
        ds.loc[ds["DIQ010"] == 1, "WTINT2YR"].sum() / ds["WTINT2YR"].sum()
        for ds in datasets
    ]
    plt.bar(years, year_diabetes)
    plt.title("Percentage of People With Diabetes by Year")
    plt.xlabel("Years 20XX")
    plt.ylabel("Percentage of People With Diabetes")
    plt.savefig(f"{output_dir}\\diabetes_by_year.png", dpi=300)
    plt.show()

    year_a1c = [
        ds.loc[ds["LBXGH"] >= 6.5, "WTMEC2YR"].sum() / ds["WTMEC2YR"].sum()
        for ds in datasets
    ]
    plt.bar(years, year_a1c)
    plt.title("Percentage of People With Diabetic Level A1C by Year")
    plt.xlabel("Years 20XX")
    plt.ylabel("Percentage of People With Diabetic Level A1C")
    plt.savefig(f"{output_dir}\\a1c_by_year.png", dpi=300)
    plt.show()

    plt.figure(figsize=(11, 6))
    race_a1c = [
        full_data.loc[
            (full_data["LBXGH"] >= 6.5) & (full_data["RIDRETH1"] == race),
            "WTMEC8YR",
        ].sum()
        / full_data.loc[full_data["RIDRETH1"] == race, "WTMEC8YR"].sum()
        for race in full_data["RIDRETH1"].unique()
    ]
    race_labels = [
        "Mexican American",
        "Other Hispanic",
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Other Race",
    ]
    plt.bar(full_data["RIDRETH1"].unique(), race_a1c)
    plt.xticks(full_data["RIDRETH1"].unique(), race_labels)
    plt.title("Percentage of People With Diabetic Level A1C by Race")
    plt.xlabel("Race")
    plt.ylabel("Percentage of People With Diabetic Level A1C")
    plt.savefig(f"{output_dir}\\a1c_by_race.png", dpi=300)
    plt.show()

    plt.figure(figsize=(11, 6))
    edu_data = full_data[
        (full_data["DMDEDUC2"].notna())
        & (
            (full_data["DMDEDUC2"].notna())
            & (full_data["DMDEDUC2"] != 9)
            & (full_data["DMDEDUC2"] != 7)
        )
    ]
    education_a1c = [
        edu_data.loc[
            (edu_data["LBXGH"] >= 6.5)
            & (edu_data["DMDEDUC2"] == education_level),
            "WTMEC8YR",
        ].sum()
        / edu_data.loc[
            edu_data["DMDEDUC2"] == education_level,
            "WTMEC8YR",
        ].sum()
        for education_level in edu_data["DMDEDUC2"].unique()
    ]
    plt.bar(edu_data["DMDEDUC2"].unique(), education_a1c)
    education_labels = [
        "Less than 9th grade",
        "9-11th grade",
        "High school graduate",
        "Some college",
        "College graduate or above",
    ]
    plt.xticks(edu_data["DMDEDUC2"].unique(), education_labels)
    plt.title(
        "Percentage of People With Diabetic Level A1C by Education Level"
    )
    plt.xlabel("Education Level")
    plt.ylabel("Percentage of People With Diabetic Level A1C")
    plt.savefig(f"{output_dir}\\a1c_by_education.png", dpi=300)
    plt.show()

    plt.close("all")
