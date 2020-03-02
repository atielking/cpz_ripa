import pandas as pd
import math

def mean(stop_data, n):
    arrest_count = 0.0
    for _, row in stop_data.iterrows():
        if row.ROS_CUSTODIAL_WITHOUT_WARRANT == 1:
            arrest_count += 1
    return arrest_count / n

# sqrt(sum((x - mu)^2) / (n - 1))
def std_dev(stop_data, n, mu):
    sigma = 0.0
    sum = 0.0
    for _, row in stop_data.iterrows():
        if row["ROS_CUSTODIAL_WITHOUT_WARRANT"] == 1:
            dif = 1.0
        else:
            dif = 0.0
        dif -= mu
        dif *= dif
        sum += dif
    sigma = math.sqrt(sum / (n - 1))
    return sigma

def perform_analysis(stop_file):

    # load the data
    stop_data = pd.read_csv(stop_file)

    # number of trials
    n = 10000

    black_count = 0.0
    white_count = 0.0

    mu = mean(stop_data, len(stop_data)) # average number of arrests
    sigma = std_dev(stop_data, len(stop_data), mu) #std dev of arrests

    # run the simulation
    for _ in range(n):
        sample = stop_data.sample(replace=False)
        if sample["ROS_CUSTODIAL_WITHOUT_WARRANT"].any() == 0:
            continue
        if sample.RACE.any() == "Black":
            black_count += 1
        else:
            white_count += 1

    z_score_black = (black_count / n - mu) / sigma
    z_score_white = (white_count / n - mu) / sigma

    # Not displaying any statistical significance thus far, though this makes
    # sense, given the small proportion of total stops resulting in a
    # specific arrest. "Big" skews are due to chance. Is this the correct approach?
    print("z score", z_score_black, z_score_white)

def main():
    perform_analysis('../ripa_sf_black_arrest.csv')
if __name__== "__main__":
    main()