import pandas as pd
import math

"""
INSTRUCTIONS TO RUN:
1) Need to run fairness.Rmd in order to generate ./RIPA_fairness_sfpd.csv file.
    This file looks at the arrests with the top 15 alleged disparities.
    TODO: this may need to be modified, as the perceived disparity may be due to
    the general small number of total arrests for that offense.

2) Need to run Fairness Dataset Generator.Rmd in order to generate the other files.
    - ../ripa_sf_black_arrest.csv - black + white arrest data for specified dept
    - ../RIPA_r_black_white.csv - black + white stop data for specified dept
    - ./CDS_Codes_non_felonies.csv' - non felony arrest codes and descriptions
        (necessary to match arrests when random sampling from stop data)
"""

# null hypothesis - There is no disparity between black and white people, results are due to chance
# alternative hypothesis - The disparity is __x for [race] people for [offense]
# ex: The disparity is 2.13x for black people for BURGLARY:SECOND DEGREE

# number arrested for offense / number stopped total
def mean(arrest_data, n, offense):
    offense_count = 0.0
    for index, row in arrest_data.iterrows():
        if row.OFFENSE == offense:
            offense_count += 1
    return offense_count / n

# sqrt(sum((x - mu)^2) / (n - 1))
def std_dev(stop_data, n, mu, offense_code):
    sigma = 0.0
    sum = 0.0
    for index, row in stop_data.iterrows():
        if row["ROS_CUSTODIAL_WITHOUT_WARRANT"] == 1:
            codes = [int(s.strip()) for s in row["ROS_CUSTODIAL_WOUT_WARRANT_CDS"].split(',')]
            if offense_code.any() in codes:
                dif = 1.0
        else:
            dif = 0.0
        dif -= mu
        dif *= dif
        sum += dif
    sigma = math.sqrt(sum / (n - 1))
    return sigma

def perform_analysis(arrest_file, stop_file, code_file, offense):

    # load the data
    arrest_data = pd.read_csv(arrest_file)
    stop_data = pd.read_csv(stop_file)
    codes = pd.read_csv(code_file)

    # FYI - could potentially be multiple codes for the same description - we
    # will account for all by using .any() whenever referencing this var
    offense_code = codes[codes["Offense.Description"] == offense]["CDS.Code"]

    # number of trials
    n = 20000

    black_count = 0.0
    white_count = 0.0

    mu = mean(arrest_data, len(stop_data), offense)
    sigma = std_dev(stop_data, len(stop_data), mu, offense_code)

    # run the simulation
    for i in range(n):
        sample = stop_data.sample(replace=False)
        if sample["ROS_CUSTODIAL_WITHOUT_WARRANT"].any() == 0:
            continue
        sample_codes = [int(s.strip()) for s in sample["ROS_CUSTODIAL_WOUT_WARRANT_CDS"].any().split(',')]
        if offense_code.any() in sample_codes:
            if sample.RACE.any() == "Black":
                black_count += 1
            else:
                white_count += 1

    z_score_black = (black_count / n - mu) / sigma
    z_score_white = (white_count / n - mu) / sigma

    # Not displaying any statistical significance thus far, though this makes
    # sense, given the small proportion of total stops resulting in a
    # specific arrest. "Big" skews are due to chance. Is this the correct approach?
    print("z score", offense, z_score_black, z_score_white)

def main():
    top_15_arrests = pd.read_csv('./RIPA_fairness_sfpd.csv')
    offenses = [row["Offense.Description"] for index, row in top_15_arrests.iterrows()]

    for offense in offenses:
        perform_analysis('../ripa_sf_black_arrest.csv', '../RIPA_r_black_white.csv', './CDS_Codes_non_felonies.csv', offense)

if __name__== "__main__":
    main()
