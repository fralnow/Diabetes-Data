# Diabetes-Data

This repository contains my CSE 163 final project analyzing trends and demographic differences in undiagnosed diabetes using data from the National Health and Nutrition Examination Survey (NHANES).

The project uses four two-year NHANES survey cycles from 2007-2008 through 2013-2014. The analysis combines demographic, diabetes questionnaire, glycohemoglobin, and body measurement data, then uses NHANES survey weights and survey-design variables to calculate weighted results and perform Rao-Scott adjusted statistical tests.

## Files

- `Analysis.py`  
  Main file for the project. It runs the final analysis, prints the numerical results, performs the survey-adjusted statistical tests, and creates the graphs used for the final report. It also calls the initialization and testing functions before running the analysis.

- `Initialization.py`  
  Loads the NHANES `.xpt` files, merges the four datasets for each survey cycle using `SEQN`, creates the diabetes classification variables, combines the four survey cycles, creates the eight-year weights, and prints summary information used during EDA.

- `NHANES_Testing.py`  
  Contains the validation tests for the project. It uses assert statements to check the required columns, unique participant IDs, survey weights, survey-design variables, diabetes classifications, combined dataset size, and whether calculated percentages are valid.

- `README.md`  
  Contains the instructions for downloading the data, installing the required libraries, and running the project.

## Required Python Libraries

The following Python libraries are required:

```text
numpy
pandas
matplotlib
seaborn
polars
scipy
statsmodels
svy
```

They can be installed from the terminal with:

```bash
pip install numpy pandas matplotlib seaborn polars scipy statsmodels svy
```

## Downloading the NHANES Data

The raw NHANES data is not included in this repository. It can be downloaded from:

https://wwwn.cdc.gov/nchs/nhanes/Default.aspx

The project uses these four survey cycles:

- 2007-2008 (`E`)
- 2009-2010 (`F`)
- 2011-2012 (`G`)
- 2013-2014 (`H`)

For each two-year survey cycle:

1. Go to the NHANES website above and click on the two-year sample you need.
2. Click **Demographics Data** and download the `.xpt` file named `DEMO`.
3. Click **Laboratory Data** and search for the file named `GHB.xpt`, then download it.
4. Click **Examination Data** and search for `BMX.xpt`, then download it.
5. Click **Questionnaire Data** and search for `DIQ.xpt`, then download it.
6. Repeat these steps for all four two-year survey cycles.

After downloading everything, there should be 16 data files. The filenames used by the code are:

```text
DEMO_E.xpt
DEMO_F.xpt
DEMO_G.xpt
DEMO_H.xpt

DIQ_E.xpt
DIQ_F.xpt
DIQ_G.xpt
DIQ_H.xpt

GHB_E.xpt
GHB_F.xpt
GHB_G.xpt
GHB_H.xpt

BMX_E.xpt
BMX_F.xpt
BMX_G.xpt
BMX_H.xpt
```

If the downloaded files use uppercase `.XPT`, either rename the extension to `.xpt` or update the filenames in `Initialization.py` to match.

## Setting the Data Path

Put all 16 `.xpt` files in the same folder.

Then open `Initialization.py` and change `base_path` near the beginning of the `initialization` function to the folder containing the downloaded files.

For example, on Windows:

```python
base_path = r"C:\Users\YourName\Downloads\NHANES_Data"
```

Do not include a filename in `base_path`; it should only point to the folder containing the `.xpt` files.

`Initialization.py` also contains an `output_dir` variable inside `plot_summary_graphs`. If you want to regenerate the EDA summary graphs from that function, change `output_dir` to the folder where you want those graphs saved.

## Running the Project

Once the libraries are installed, the 16 data files are downloaded, and `base_path` has been updated, open a terminal in the repository folder and run:

```bash
python Analysis.py
```

`Analysis.py` will:

1. Load and merge the NHANES datasets using `Initialization.py`.
2. Run the validation tests from `NHANES_Testing.py`.
3. Print `All data assertions passed.` if the validation tests succeed.
4. Run the final analysis.
5. Print the weighted percentages, population estimates, contingency-table information, and Rao-Scott statistical-test results to the terminal.
6. Save the final analysis graphs as `.png` files in the directory where the program is run.

The validation tests are called automatically when `Analysis.py` is run, so there is no separate testing command required.

## Reproducing the EDA Summary Graphs

The main final-analysis results are produced by `Analysis.py`. `Initialization.py` also contains the `plot_summary_graphs` function used for additional EDA graphs.

After setting both `base_path` and `output_dir`, these can be regenerated from a Python interpreter with:

```python
from Initialization import initialization, plot_summary_graphs

datasets, full_data = initialization()
plot_summary_graphs(datasets, full_data)
```

## Notes

- The raw NHANES `.xpt` files are intentionally not stored in this repository.
- All four datasets for a survey cycle are joined using the participant identifier `SEQN`.
- The final analysis uses four NHANES cycles from 2007-2008 through 2013-2014.
- `Undiagnosed_Status == 1` represents participants who had an HbA1c value of at least 6.5% but did not report a previous diabetes diagnosis.
