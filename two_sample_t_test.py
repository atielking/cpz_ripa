import pandas as pd
import math

from statsmodels.stats.power import TTestIndPower
import numpy as np
from scipy import stats

# References:
# https://towardsdatascience.com/master-your-hypothesis-test-a-tutorial-on-power-bootstrapping-sample-selection-and-outcome-273d6739d3e5
# https://www.statisticshowto.datasciencecentral.com/cohens-d/
# https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Two-Sample_T-Test.pdf

def clean_data(filename):
    df = pd.read_csv(filename)
    overall_hispanic = df[df['RACE'].str.contains("Hispanic")==True]
    overall_white = df[df['RACE'].str.contains("White")==True]

    #print(overall_black, overall_white)

    hispanic_df = overall_hispanic['ROS_CUSTODIAL_WITHOUT_WARRANT']
    hispanic_arr = np.array(hispanic_df)
    white_df = overall_white['ROS_CUSTODIAL_WITHOUT_WARRANT']
    white_arr = np.array(white_df)

    return (hispanic_arr, white_arr)

def calc_cohen_d(group1, group2):
    """
    Calculate and Define Cohen's d.
    Cohen's d effect size measures the difference between the means of both groups.
    This gives some insight into how much change is actually occurring between the two groups.
    The outcome of the Cohen's d effect size is measured in standard deviations.

    0.0-0.20 = small effect
    0.20-0.50 = medium effect
    0.50+ = large effect
    """
    # group1: Series or NumPy array
    # group2: Series or NumPy array
    # returns a floating point number

    diff = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)

    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    #print("Cohen's D", abs(d))
    return d

def calc_min_sample_size(group1, group2, cohens_d):
    effect = cohens_d
    alpha = 0.05
    power = 1

    # sample 2 / sample 1
    ratio = len(group1) / len(group2)

    # Perform power analysis
    analysis = TTestIndPower()
    result = analysis.solve_power(effect, power=power, nobs1=None,ratio=ratio, alpha=alpha)
    #print("The minimum sample size:", result)
    #print("Number of black stopped:", len(group1))
    #print("Number of white stopped:", len(group2))
    return int(result)

def perform_two_sample_t_test(group1, group2):
    cohens_d = calc_cohen_d(group1, group2)
    sample_size = calc_min_sample_size(group1, group2, cohens_d)

    # Bootstrapping methodology
    sample_means_overall_group1 = []
    for _ in range(100):
        sample_mean = np.random.choice(group1,size=sample_size).mean()
        sample_means_overall_group1.append(sample_mean)

    sample_means_overall_group2 = []
    for _ in range(100):
        sample_mean = np.random.choice(group2,size=sample_size).mean()
        sample_means_overall_group2.append(sample_mean)

    t_stat = stats.ttest_ind(sample_means_overall_group1, sample_means_overall_group2)
    print(t_stat)

    # Analyze results
    msg = ""
    if (t_stat.pvalue < 0.025):
        msg = '''
    Since our p-value is less than 0.025, we are going to accept our alternative
    hypothesis and conclude that there is a difference in the proportion of
    black people arrested and white people arrested.
        '''
    else:
        msg = '''
    Since our p-value is not less than than 0.025, we are going to accept our
    null hypothesis and conclude there is no difference in the proportion
    of black people arrested and white people arrested - any difference
    seen is due to chance.
        '''
    print(msg)

def main():
    intro = '''
    Null: There is not a statistical difference between the number of black arrested
    out of the number of black people stopped and the number of white arrested out of
    the number of white people stopped.

    Alt: There is a statistical difference between the proportion of black people arrested
    after being stopped and the proportion of white people arrested after being stopped.

    We will define our alpha as 0.05 (typical for a two-tailed t-test)
    This means that we expect that given our null hypothesis is true that our test
    will be correct 95% of the time where the other 5% is incorrect because of
    random chance. Since this is a two-tailed test, we will
    split the alpha error in half and put it on both sides of our distribution.
    This means that the p-value we are looking for is .025, so 2.5 percent.
    If our p-value is less than or equal to our anticipated error then we would
    reject our Null hypothesis.
    '''
    print(intro)
    hispanic_arr, white_arr = clean_data('../arrest_sig/ripa_sf_hispanic_arrest.csv')
    perform_two_sample_t_test(hispanic_arr, white_arr)

if __name__== "__main__":
    main()
