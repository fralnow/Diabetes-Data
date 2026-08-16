"""
Faris Alnowami
CSE 163 AC
Final Project

This module contains testing and validation functions for the NHANES
dataset analysis. It includes data validation checks to ensure datasets
meet expected criteria and integrity constraints.
"""

import pandas as pd


def validation(
    datasets: list[pd.DataFrame],
    full_data: pd.DataFrame,
) -> None:
    """
    Given a list of the yearly datasets and the combined dataset, performs
    validation checks to ensure that the datasets meet the expected criteria.
    """
    needed_columns = [
        'SEQN', 'SDDSRVYR', 'DIQ010', 'RIAGENDR', 'RIDAGEYR',
        'RIDRETH1', 'DMDEDUC2', 'WTINT2YR', 'WTMEC2YR', 'LBXGH',
        'A1C_Based_Diabetes', 'Undiagnosed_Status', 'BMXBMI',
        'SDMVSTRA', 'SDMVPSU'
    ]
    data_E, data_F, data_G, data_H = datasets
    for i, ds in enumerate(datasets):
        assert set(ds.columns) == set(needed_columns)
        assert ds['SEQN'].is_unique
        assert ds['SEQN'].notna().all()
        assert ds['WTINT2YR'].notna().all()
        assert ds['WTMEC2YR'].notna().all()
        assert ds["SDMVSTRA"].notna().all()
        assert ds["SDMVPSU"].notna().all()
        assert len(ds) > 0
        assert ds.loc[ds["LBXGH"] >= 6.5, "A1C_Based_Diabetes"].eq(1).all()
        assert ds.loc[ds["LBXGH"] < 6.5, "A1C_Based_Diabetes"].eq(2).all()
        assert ds.loc[
            (ds["LBXGH"] >= 6.5) & (ds["DIQ010"] == 2),
            "Undiagnosed_Status",
        ].eq(1).all()
    sqn_E = set(data_E["SEQN"])
    sqn_F = set(data_F["SEQN"])
    sqn_G = set(data_G["SEQN"])
    sqn_H = set(data_H["SEQN"])
    assert (sqn_E & sqn_F & sqn_G & sqn_H) == set()

    assert len(full_data) == sum(len(ds) for ds in datasets)
    assert full_data['SEQN'].is_unique
    assert full_data['SEQN'].notna().all()
    assert (full_data['WTINT2YR'].fillna(0) >= 0).all()
    assert (full_data['WTMEC2YR'].fillna(0) >= 0).all()
    assert ('WTINT8YR' in full_data.columns and
            'WTMEC8YR' in full_data.columns)

    for col in ["RIDRETH1", "DMDEDUC2", "RIAGENDR"]:
        if col in full_data.columns:
            for value in full_data[col].unique():
                if pd.notna(value):
                    subset = full_data[full_data[col] == value]
                    undiag_sum = (
                        subset[subset["Undiagnosed_Status"] == 1]["WTMEC8YR"]
                        .sum()
                    )
                    total_sum = subset["WTMEC8YR"].sum()
                    pct_undiagnosed = undiag_sum / total_sum * 100
                    assert 0 <= pct_undiagnosed <= 100, (
                        f"Undiagnosed percentage for {col}={value} "
                        f"is {pct_undiagnosed}, should be 0-100"
                    )

    print("All data assertions passed.")
