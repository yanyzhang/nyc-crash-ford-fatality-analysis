### Import packages in the virtual environment
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats as scipy
from scipy.stats import shapiro
import numpy as np
import sys
import os

os.chdir(r"C:\Documents\Courses\Data Mining\Project")
## ⚠️ CHANGE THIS to your own working folder path

pd.set_option('display.max_columns', 500)

## Load cleaned data
mydata = pd.read_csv("cleaned_collisions.csv")
if 'Fatality' not in mydata.columns and 'KILLED' in mydata.columns:
    mydata = mydata.rename(columns={'KILLED': 'Fatality'})

## Load original data for Raw Comparison by Make
original = pd.read_csv("Motor_Vehicle_Collisions_-_Crashes.csv", low_memory=False)

file = open('output.txt', 'wt')
sys.stdout = file

print(mydata.describe())

df = pd.DataFrame(mydata)

## Create VEHICLE_AGE: years the vehicle has been used before the accident
## Calculated as crash year minus manufacturing year
df['CRASH_DATE'] = pd.to_datetime(original.loc[mydata.index, 'CRASH_DATE'], errors='coerce')
df['CRASH_YEAR'] = df['CRASH_DATE'].dt.year
df['VEHICLE_AGE'] = df['CRASH_YEAR'] - df['VEHICLE_YEAR']
## Drop CRASH_DATE and CRASH_YEAR after use — not needed as model variables
df = df.drop(columns=['CRASH_DATE', 'CRASH_YEAR'])
## A higher value means an older vehicle with more years of use before the crash


######################## Summary Statistics: Key Variables

print("\n=== Summary Statistics: Key Variables ===")
print(df[['VEHICLE_AGE',
          'VEHICLE_OCCUPANTS',
          'FORD',
          'Fatality',
          'VEHICLE_TYPE_Pick-up Truck',
          'VEHICLE_TYPE_Motorcycle',
          'VEHICLE_TYPE_Tractor Truck Diesel',
          'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle'
          ]].describe().round(4))

print("\n=== Key Counts ===")
print(f"Total observations:              {len(df):,}")
print(f"Fatal crashes (Fatality = 1):    {df['Fatality'].sum():,}  ({df['Fatality'].mean()*100:.4f}%)")
print(f"Ford vehicles (FORD = 1):        {df['FORD'].sum():,}  ({df['FORD'].mean()*100:.2f}%)")
print(f"Male drivers:                    {df['DRIVER_SEX_M'].sum():,}  ({df['DRIVER_SEX_M'].mean()*100:.2f}%)")
print(f"Unlicensed drivers:              {df['DRIVER_LICENSE_STATUS_Unlicensed'].sum():,}  ({df['DRIVER_LICENSE_STATUS_Unlicensed'].mean()*100:.2f}%)")
print(f"Pick-up Trucks:                  {df['VEHICLE_TYPE_Pick-up Truck'].sum():,}  ({df['VEHICLE_TYPE_Pick-up Truck'].mean()*100:.2f}%)")
print(f"Motorcycles:                     {df['VEHICLE_TYPE_Motorcycle'].sum():,}  ({df['VEHICLE_TYPE_Motorcycle'].mean()*100:.2f}%)")
print(f"Tractor Trucks:                  {df['VEHICLE_TYPE_Tractor Truck Diesel'].sum():,}  ({df['VEHICLE_TYPE_Tractor Truck Diesel'].mean()*100:.2f}%)")
print(f"SUV/Station Wagon:               {df['VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle'].sum():,}  ({df['VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle'].mean()*100:.2f}%)")


######################## Raw Comparison by Make (Summary Statistics)

## Merge original VEHICLE_MAKE back into cleaned data using index alignment
orig_make = original.loc[mydata.index, ['VEHICLE_MAKE']].copy()
orig_make['Fatality'] = mydata['Fatality'].values

## Keep top 10 most common makes
top10_makes = orig_make['VEHICLE_MAKE'].value_counts().nlargest(10).index
df_top = orig_make[orig_make['VEHICLE_MAKE'].isin(top10_makes)]

## Calculate fatality rate per make
comparison = df_top.groupby('VEHICLE_MAKE').agg(
    Observations=('Fatality', 'count'),
    Fatal_Crashes=('Fatality', 'sum')
).reset_index()
comparison['Fatality_Rate_%'] = (
    comparison['Fatal_Crashes'] / comparison['Observations'] * 100
).round(4)
comparison = comparison.sort_values('Fatality_Rate_%', ascending=False)

print("\n=== Raw Comparison by Vehicle Make ===")
print(comparison.to_string(index=False))


## Bubble chart: Fatality Rate vs Vehicle Volume by Make
fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f9fa')

for _, row in comparison.iterrows():
    is_ford = 'FORD' in row['VEHICLE_MAKE']
    color = '#c0392b' if is_ford else '#2c7bb6'
    alpha = 0.9 if is_ford else 0.65
    size = row['Observations'] / 8

    ax.scatter(row['Observations'], row['Fatality_Rate_%'],
               s=size, color=color, alpha=alpha, zorder=3,
               edgecolors='white', linewidth=1.5 if is_ford else 0.8)

    offset_y = 0.004 if not is_ford else 0.007
    ax.text(row['Observations'], row['Fatality_Rate_%'] + offset_y,
            row['VEHICLE_MAKE'].replace(' -CAR/SUV', ''),
            ha='center', va='bottom',
            color='#c0392b' if is_ford else '#333333',
            fontsize=10 if is_ford else 9,
            fontweight='bold' if is_ford else 'normal')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.xaxis.grid(True, color='#dddddd', linewidth=0.8, zorder=0)
ax.yaxis.grid(True, color='#dddddd', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors='#555555', labelsize=9)
ax.set_xlabel('Number of Observations (Vehicle Count)', color='#555555', fontsize=11, labelpad=10)
ax.set_ylabel('Fatality Rate (%)', color='#555555', fontsize=11, labelpad=10)
ax.set_title('Fatality Rate vs Vehicle Volume by Make',
             color='#1a1a1a', fontsize=14, fontweight='bold', pad=15)
fig.text(0.5, 0.93, 'Bubble size represents number of observations  |  Ford highlighted in red',
         ha='center', color='#888888', fontsize=9)

ford_patch = mpatches.Patch(color='#c0392b', label='Ford')
other_patch = mpatches.Patch(color='#2c7bb6', label='Other Brands')
ax.legend(handles=[ford_patch, other_patch],
          facecolor='white', edgecolor='#cccccc',
          labelcolor='#333333', fontsize=10, loc='upper right', framealpha=0.9)

ford_row = comparison[comparison['VEHICLE_MAKE'].str.contains('FORD')].iloc[0]
ax.annotate('Ford ranks 5th\namong 10 brands',
            xy=(ford_row['Observations'], ford_row['Fatality_Rate_%']),
            xytext=(ford_row['Observations'] + 1200, ford_row['Fatality_Rate_%'] - 0.02),
            color='#c0392b', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdf0ef', edgecolor='#c0392b', alpha=0.8))

plt.tight_layout()
plt.savefig('fatality_bubble.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


######################## Reference Model (FORD only — kept for code reference, not included in output)

x = df['FORD']
## Define the independent variable: Ford dummy (1 = Ford, 0 = other brand)
y = df['Fatality']
## Define the dependent variable: death dummy (1 = someone died, 0 = no death)

x = sm.add_constant(x)
## Add a constant in the regression model

model1 = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
## method='bfgs' and maxiter=200 ensure the model converges properly
results_model1 = model1.summary()  ## kept for reference
#print("\n=== Model 1: FORD only ===")  ## excluded from output
#print(results_model1)  ## excluded from output


######################## Model 1: FORD + Controls (Main Model)

x = df[['FORD',
        'VEHICLE_AGE',
        ## VEHICLE_AGE = 2016 - VEHICLE_YEAR: years of vehicle use before accident
        'VEHICLE_OCCUPANTS',
        'DRIVER_SEX_M',
        'DRIVER_LICENSE_STATUS_Unlicensed',
        ## Note: DRIVER_LICENSE_STATUS_Permit removed - zero death cases cause separation
        'VEHICLE_TYPE_Pick-up Truck',
        'VEHICLE_TYPE_Motorcycle',
        'VEHICLE_TYPE_Tractor Truck Diesel',
        'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle',
        'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction',
        'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way',
        'CONTRIBUTING_FACTOR_1_Unspecified',
        'TIME_OF_DAY_Evening',
        'TIME_OF_DAY_Morning',
        'TIME_OF_DAY_Night'
        ]]
## Define the independent variables:
## - FORD: main variable of interest
## - VEHICLE_AGE: years of vehicle use before accident (older = more years used)
## - VEHICLE_OCCUPANTS: number of people in car
## - DRIVER_SEX_M: male driver dummy
## - DRIVER_LICENSE_STATUS_Unlicensed: unlicensed driver dummy
## - VEHICLE_TYPE: type of vehicle (controlling for vehicle class)
## - CONTRIBUTING_FACTOR_1: main cause of accident
## - TIME_OF_DAY: morning/evening/night vs afternoon (reference category)
y = df['Fatality']

x = sm.add_constant(x)

model2 = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
results_model2 = model2.summary()
print("\n=== Model 1: FORD + Controls ===")
print(results_model2)


######################## Odds Ratio — Model 1: FORD + Controls

print("\n=== Odds Ratio: Model 1: FORD + Controls ===")
odds_ratio = pd.DataFrame({
    'Odds Ratio': np.exp(model2.params),
    'Lower 95% CI': np.exp(model2.conf_int()[0]),
    'Upper 95% CI': np.exp(model2.conf_int()[1]),
    'P-value': model2.pvalues
}).round(4)
print(odds_ratio.to_string())
print("\nNote: Odds Ratio > 1 means higher fatality risk.")
print("      FORD Odds Ratio interpreted as: Ford vehicles have X times")
print("      the odds of fatal crash compared to non-Ford vehicles.")


######################## Odds Ratio Forest Plot

or_df = pd.DataFrame({
    'OR': np.exp(model2.params),
    'Lower': np.exp(model2.conf_int()[0]),
    'Upper': np.exp(model2.conf_int()[1]),
    'pvalue': model2.pvalues
}).drop('const').reset_index()
or_df.columns = ['Variable', 'OR', 'Lower', 'Upper', 'pvalue']

name_map = {
    'FORD': 'Ford',
    'VEHICLE_AGE': 'Vehicle Age',
    'VEHICLE_OCCUPANTS': 'Vehicle Occupants',
    'DRIVER_SEX_M': 'Male Driver',
    'DRIVER_LICENSE_STATUS_Unlicensed': 'Unlicensed Driver',
    'VEHICLE_TYPE_Pick-up Truck': 'Pick-up Truck',
    'VEHICLE_TYPE_Motorcycle': 'Motorcycle',
    'VEHICLE_TYPE_Tractor Truck Diesel': 'Tractor Truck',
    'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle': 'SUV/Station Wagon',
    'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction': 'Driver Inattention',
    'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way': 'Failure to Yield',
    'CONTRIBUTING_FACTOR_1_Unspecified': 'Unspecified Factor',
    'TIME_OF_DAY_Evening': 'Evening',
    'TIME_OF_DAY_Morning': 'Morning',
    'TIME_OF_DAY_Night': 'Night'
}
or_df['Variable'] = or_df['Variable'].map(name_map)
or_df = or_df.sort_values('OR', ascending=True)

fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f9fa')

for i, row in or_df.iterrows():
    is_ford = row['Variable'] == 'Ford'
    is_sig = row['pvalue'] < 0.05
    color = '#c0392b' if is_ford else ('#2c7bb6' if is_sig else '#aaaaaa')
    ax.plot([row['Lower'], row['Upper']], [i, i],
            color=color, linewidth=2, zorder=2)
    ax.scatter(row['OR'], i, color=color,
               s=100 if is_ford else 60, zorder=3,
               edgecolors='white', linewidth=0.8)

ax.axvline(x=1, color='#333333', linewidth=1.2, linestyle='--', zorder=1, alpha=0.7)
ax.set_yticks(range(len(or_df)))
ax.set_yticklabels(or_df['Variable'], fontsize=10)

for label, (_, row) in zip(ax.get_yticklabels(), or_df.iterrows()):
    if row['Variable'] == 'Ford':
        label.set_color('#c0392b')
        label.set_fontweight('bold')
    elif row['pvalue'] < 0.05:
        label.set_color('#2c7bb6')
    else:
        label.set_color('#888888')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.xaxis.grid(True, color='#dddddd', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis='x', colors='#555555', labelsize=9)
ax.tick_params(axis='y', length=0)
ax.set_xscale('log')
ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20, 50])
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
ax.set_xlabel('Odds Ratio — values above 1 indicate higher fatality risk\n(log scale, 95% CI shown)',
              color='#555555', fontsize=10, labelpad=10)
ax.set_title('Odds Ratio — Model 1: FORD + Controls',
             color='#1a1a1a', fontsize=13, fontweight='bold', pad=15)
fig.text(0.5, 0.93,
         'Red = Ford  |  Blue = significant (p < 0.05)  |  Grey = not significant  |  Dashed line = OR of 1 (no effect)',
         ha='center', color='#888888', fontsize=9)

plt.tight_layout()
plt.savefig('odds_ratio_forest.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


######################## Predicted Probability Plot

## Define baseline scenario: average values, male driver, afternoon
base = {
    'const': 1,
    'FORD': 0,
    'VEHICLE_AGE': df['VEHICLE_AGE'].mean(),
    ## VEHICLE_AGE replaces VEHICLE_YEAR — average years of use
    'VEHICLE_OCCUPANTS': df['VEHICLE_OCCUPANTS'].mean(),
    'DRIVER_SEX_M': 1,
    'DRIVER_LICENSE_STATUS_Unlicensed': 0,
    'VEHICLE_TYPE_Pick-up Truck': 0,
    'VEHICLE_TYPE_Motorcycle': 0,
    'VEHICLE_TYPE_Tractor Truck Diesel': 0,
    'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle': 0,
    'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction': 0,
    'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way': 0,
    'CONTRIBUTING_FACTOR_1_Unspecified': 1,
    'TIME_OF_DAY_Evening': 0,
    'TIME_OF_DAY_Morning': 0,
    'TIME_OF_DAY_Night': 0
}

scenarios = {
    'Non-Ford\nAfternoon': {**base, 'FORD': 0},
    'Ford\nAfternoon':     {**base, 'FORD': 1},
    'Non-Ford\nNight':     {**base, 'FORD': 0, 'TIME_OF_DAY_Night': 1},
    'Ford\nNight':         {**base, 'FORD': 1, 'TIME_OF_DAY_Night': 1},
    'Non-Ford\nSUV':       {**base, 'FORD': 0, 'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle': 1},
    'Ford\nSUV':           {**base, 'FORD': 1, 'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle': 1},
    'Non-Ford\nUnlicensed':{**base, 'FORD': 0, 'DRIVER_LICENSE_STATUS_Unlicensed': 1},
    'Ford\nUnlicensed':    {**base, 'FORD': 1, 'DRIVER_LICENSE_STATUS_Unlicensed': 1},
}

probs = {}
for label, scenario in scenarios.items():
    xvals    = pd.Series(scenario)
    log_odds = model2.params.dot(xvals)
    prob     = 1 / (1 + np.exp(-log_odds))
    probs[label] = prob * 100

labels = list(probs.keys())
values = list(probs.values())
colors = ['#c0392b' if 'Ford' in l and 'Non' not in l else '#2c7bb6' for l in labels]

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f9fa')

bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3,
              edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}%', ha='center', va='bottom', fontsize=9,
            color='#333333', fontweight='bold')

pairs       = [(0,1), (2,3), (4,5), (6,7)]
pair_labels = ['Afternoon', 'Night', 'SUV', 'Unlicensed']
for (i, j), lbl in zip(pairs, pair_labels):
    y_top = max(values[i], values[j]) + 0.008
    ax.plot([i, i, j, j], [y_top]*4, color='#aaaaaa', lw=0.8)
    ax.text((i+j)/2, y_top + 0.004, lbl,
            ha='center', fontsize=9, color='#555555')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.yaxis.grid(True, color='#dddddd', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(colors='#555555', labelsize=9)
ax.set_ylabel('Predicted Probability of Fatality (%)',
              color='#555555', fontsize=11, labelpad=10)
ax.set_title('Predicted Probability of Fatality: Ford vs Non-Ford by Condition',
             color='#1a1a1a', fontsize=13, fontweight='bold', pad=15)
fig.text(0.5, 0.93,
         'All other variables held at sample mean  |  Male driver, average vehicle age',
         ha='center', color='#888888', fontsize=9)

ford_patch  = mpatches.Patch(color='#c0392b', label='Ford')
other_patch = mpatches.Patch(color='#2c7bb6', label='Non-Ford')
ax.legend(handles=[ford_patch, other_patch],
          facecolor='white', edgecolor='#cccccc',
          labelcolor='#333333', fontsize=10, loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('predicted_prob.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


######################## Predicted Probability Output

print("\n=== Predicted Probability of Fatality: Ford vs Non-Ford ===")
print(f"{'Condition':<30} {'Non-Ford':>12} {'Ford':>12} {'Difference':>12}")
print("-" * 68)

condition_pairs = [
    ('Afternoon (baseline)',
     {**base, 'FORD': 0},
     {**base, 'FORD': 1}),
    ('Night',
     {**base, 'FORD': 0, 'TIME_OF_DAY_Night': 1},
     {**base, 'FORD': 1, 'TIME_OF_DAY_Night': 1}),
    ('Evening',
     {**base, 'FORD': 0, 'TIME_OF_DAY_Evening': 1},
     {**base, 'FORD': 1, 'TIME_OF_DAY_Evening': 1}),
    ('Morning',
     {**base, 'FORD': 0, 'TIME_OF_DAY_Morning': 1},
     {**base, 'FORD': 1, 'TIME_OF_DAY_Morning': 1}),
    ('SUV',
     {**base, 'FORD': 0, 'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle': 1},
     {**base, 'FORD': 1, 'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle': 1}),
    ('Pick-up Truck',
     {**base, 'FORD': 0, 'VEHICLE_TYPE_Pick-up Truck': 1},
     {**base, 'FORD': 1, 'VEHICLE_TYPE_Pick-up Truck': 1}),
    ('Unlicensed Driver',
     {**base, 'FORD': 0, 'DRIVER_LICENSE_STATUS_Unlicensed': 1},
     {**base, 'FORD': 1, 'DRIVER_LICENSE_STATUS_Unlicensed': 1}),
    ('Female Driver',
     {**base, 'FORD': 0, 'DRIVER_SEX_M': 0},
     {**base, 'FORD': 1, 'DRIVER_SEX_M': 0}),
]

for condition, non_ford_scenario, ford_scenario in condition_pairs:
    xvals_nf = pd.Series(non_ford_scenario)
    xvals_f  = pd.Series(ford_scenario)
    prob_nf  = 1 / (1 + np.exp(-model2.params.dot(xvals_nf))) * 100
    prob_f   = 1 / (1 + np.exp(-model2.params.dot(xvals_f)))  * 100
    diff     = prob_f - prob_nf
    print(f"{condition:<30} {prob_nf:>11.4f}% {prob_f:>11.4f}% {diff:>+11.4f}%")

print("-" * 68)
print("Note: All other variables held at sample mean.")
print("      Baseline = Male driver, Afternoon, average vehicle age.")


######################## Sub-Sample 1 Model: Logistic regression with Vehicle Type

## SUV/Truck only
suv_truck_cols = ['VEHICLE_TYPE_Pick-up Truck',
                  'VEHICLE_TYPE_Tractor Truck Diesel',
                  'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle']
df_suvtruck = df[(df[suv_truck_cols].sum(axis=1)) == 1]

## Non SUV/Truck — Motorcycle excluded (coef = 3.45) to avoid overshadowing Ford effect
df_other = df[(df[suv_truck_cols].sum(axis=1)) == 0]
df_other = df_other[df_other['VEHICLE_TYPE_Motorcycle'] == 0]

## Vehicle type dummies excluded — sub-sample already restricted to one vehicle category
x = df_suvtruck[['FORD', 'VEHICLE_AGE', 'VEHICLE_OCCUPANTS', 'DRIVER_SEX_M',
                  'DRIVER_LICENSE_STATUS_Unlicensed',
                  'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction',
                  'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way',
                  'CONTRIBUTING_FACTOR_1_Unspecified',
                  'TIME_OF_DAY_Evening', 'TIME_OF_DAY_Morning', 'TIME_OF_DAY_Night']]
y = df_suvtruck['Fatality']
x = sm.add_constant(x)
model_suvtruck = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
results_model_suvtruck = model_suvtruck.summary()
print("\n=== Sub-Sample 1a: SUV/Truck Only ===")
print(results_model_suvtruck)

x = df_other[['FORD', 'VEHICLE_AGE', 'VEHICLE_OCCUPANTS', 'DRIVER_SEX_M',
              'DRIVER_LICENSE_STATUS_Unlicensed',
              'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction',
              'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way',
              'CONTRIBUTING_FACTOR_1_Unspecified',
              'TIME_OF_DAY_Evening', 'TIME_OF_DAY_Morning', 'TIME_OF_DAY_Night']]
y = df_other['Fatality']
x = sm.add_constant(x)
model_other = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
results_model_other = model_other.summary()
print("\n=== Sub-Sample 1b: Non SUV/Truck ===")
print(results_model_other)


######################## Sub-Sample 2 Model: Logistic regression with Gender

## DRIVER_SEX_M excluded — sub-sample already restricted to one gender
df_male   = df[df['DRIVER_SEX_M'] == 1]
df_female = df[df['DRIVER_SEX_M'] == 0]

x = df_male[['FORD', 'VEHICLE_AGE', 'VEHICLE_OCCUPANTS',
             'DRIVER_LICENSE_STATUS_Unlicensed',
             'VEHICLE_TYPE_Pick-up Truck', 'VEHICLE_TYPE_Motorcycle',
             'VEHICLE_TYPE_Tractor Truck Diesel',
             'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle',
             'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction',
             'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way',
             'CONTRIBUTING_FACTOR_1_Unspecified',
             'TIME_OF_DAY_Evening', 'TIME_OF_DAY_Morning', 'TIME_OF_DAY_Night']]
y = df_male['Fatality']
x = sm.add_constant(x)
model_male = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
results_model_male = model_male.summary()
print("\n=== Sub-Sample 2a: Male Drivers Only ===")
print(results_model_male)

x = df_female[['FORD', 'VEHICLE_AGE', 'VEHICLE_OCCUPANTS',
               'DRIVER_LICENSE_STATUS_Unlicensed',
               'VEHICLE_TYPE_Pick-up Truck', 'VEHICLE_TYPE_Motorcycle',
               'VEHICLE_TYPE_Tractor Truck Diesel',
               'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle',
               'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction',
               'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way',
               'CONTRIBUTING_FACTOR_1_Unspecified',
               'TIME_OF_DAY_Evening', 'TIME_OF_DAY_Morning', 'TIME_OF_DAY_Night']]
y = df_female['Fatality']
x = sm.add_constant(x)
model_female = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
results_model_female = model_female.summary()
print("\n=== Sub-Sample 2b: Female Drivers Only ===")
print(results_model_female)


######################## Sub-Sample 3 Model: Logistic regression with Vehicle Age

## Split by VEHICLE_AGE (years used before accident)
## New:  VEHICLE_AGE <= 5  (vehicles 0-5 years old, manufactured 2011-2016)
## Mid:  VEHICLE_AGE 6-15  (vehicles 6-15 years old, manufactured 2001-2010)
## Old:  VEHICLE_AGE > 15  (vehicles over 15 years old, manufactured before 2001)
df_new = df[df['VEHICLE_AGE'] <= 5]
df_mid = df[(df['VEHICLE_AGE'] > 5) & (df['VEHICLE_AGE'] <= 15)]
df_old = df[df['VEHICLE_AGE'] > 15]

## VEHICLE_AGE excluded — sub-sample already restricted to one age group
x = df_new[['FORD', 'VEHICLE_OCCUPANTS', 'DRIVER_SEX_M',
            'DRIVER_LICENSE_STATUS_Unlicensed',
            'VEHICLE_TYPE_Pick-up Truck', 'VEHICLE_TYPE_Motorcycle',
            'VEHICLE_TYPE_Tractor Truck Diesel',
            'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle',
            'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction',
            'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way',
            'CONTRIBUTING_FACTOR_1_Unspecified',
            'TIME_OF_DAY_Evening', 'TIME_OF_DAY_Morning', 'TIME_OF_DAY_Night']]
y = df_new['Fatality']
x = sm.add_constant(x)
model_new = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
results_model_new = model_new.summary()
print("\n=== Sub-Sample 3a: New Vehicles (Age 0-5 years, 2011-2016) ===")
print(results_model_new)

x = df_mid[['FORD', 'VEHICLE_OCCUPANTS', 'DRIVER_SEX_M',
            'DRIVER_LICENSE_STATUS_Unlicensed',
            'VEHICLE_TYPE_Pick-up Truck', 'VEHICLE_TYPE_Motorcycle',
            'VEHICLE_TYPE_Tractor Truck Diesel',
            'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle',
            'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction',
            'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way',
            'CONTRIBUTING_FACTOR_1_Unspecified',
            'TIME_OF_DAY_Evening', 'TIME_OF_DAY_Morning', 'TIME_OF_DAY_Night']]
y = df_mid['Fatality']
x = sm.add_constant(x)
model_mid = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
results_model_mid = model_mid.summary()
print("\n=== Sub-Sample 3b: Mid Vehicles (Age 6-15 years, 2001-2010) ===")
print(results_model_mid)

x = df_old[['FORD', 'VEHICLE_OCCUPANTS', 'DRIVER_SEX_M',
            'DRIVER_LICENSE_STATUS_Unlicensed',
            'VEHICLE_TYPE_Pick-up Truck', 'VEHICLE_TYPE_Motorcycle',
            'VEHICLE_TYPE_Tractor Truck Diesel',
            'VEHICLE_TYPE_Station Wagon/Sport Utility Vehicle',
            'CONTRIBUTING_FACTOR_1_Driver Inattention/Distraction',
            'CONTRIBUTING_FACTOR_1_Failure to Yield Right-of-Way',
            'CONTRIBUTING_FACTOR_1_Unspecified',
            'TIME_OF_DAY_Evening', 'TIME_OF_DAY_Morning', 'TIME_OF_DAY_Night']]
y = df_old['Fatality']
x = sm.add_constant(x)
model_old = sm.Logit(y, x).fit(method='bfgs', maxiter=200)
results_model_old = model_old.summary()
print("\n=== Sub-Sample 3c: Old Vehicles (Age > 15 years, before 2001) ===")
print(results_model_old)

#### End of the Models ####


######################## Ford Effect Heatmap — Summary Across All Models

heatmap_rows = []
all_models_hm = [
    ('Main\nModel',           model2,        len(df)),
    ('Sub 1a\nSUV/Truck',     model_suvtruck, len(df_suvtruck)),
    ('Sub 1b\nNon-SUV',       model_other,   len(df_other)),
    ('Sub 2a\nMale',          model_male,    len(df_male)),
    ('Sub 2b\nFemale',        model_female,  len(df_female)),
    ('Sub 3a\nVehicle Age 0-5 yrs',           model_new,     len(df_new)),
    ('Sub 3b\nVehicle Age 6-15 yrs',           model_mid,     len(df_mid)),
    ('Sub 3c\nVehicle Age >15 yrs',           model_old,     len(df_old)),
]
for name, model, n in all_models_hm:
    heatmap_rows.append({
        'Model':       name,
        'Ford Coef':   round(model.params['FORD'], 3),
        'P-value':     round(model.pvalues['FORD'], 3),
        'Fatal Cases': int(model.model.endog.sum()),
        'N':           n
    })
hm = pd.DataFrame(heatmap_rows)

coef_cmap = plt.cm.RdYlGn_r
pval_cmap = plt.cm.RdYlGn
n_models  = len(hm)
y_positions = [3.2, 2.1, 1.0, 0.0]
cell_h = 0.85

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for col_i, (_, row) in enumerate(hm.iterrows()):
    cn = np.clip((row['Ford Coef']-(-1))/(3-(-1)), 0, 1)
    ax.add_patch(plt.Rectangle([col_i, y_positions[0]], 1, cell_h,
                 color=coef_cmap(cn), zorder=1))
    ax.text(col_i+0.5, y_positions[0]+cell_h/2, f'{row["Ford Coef"]:.3f}',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white' if cn>0.65 or cn<0.25 else '#333333')

    pn = np.clip(1-row['P-value'], 0, 1)
    ax.add_patch(plt.Rectangle([col_i, y_positions[1]], 1, cell_h,
                 color=pval_cmap(pn), zorder=1))
    sig = ' *' if row['P-value'] < 0.05 else ''
    ax.text(col_i+0.5, y_positions[1]+cell_h/2, f'{row["P-value"]:.3f}{sig}',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white' if pn>0.7 else '#333333')

    ax.add_patch(plt.Rectangle([col_i, y_positions[2]], 1, cell_h,
                 color='#f0f4f8', zorder=1, ec='#dddddd', lw=0.5))
    ax.text(col_i+0.5, y_positions[2]+cell_h/2, str(row['Fatal Cases']),
            ha='center', va='center', fontsize=10, color='#333333')

    ax.add_patch(plt.Rectangle([col_i, y_positions[3]], 1, cell_h,
                 color='#f8f8f8', zorder=1, ec='#dddddd', lw=0.5))
    ax.text(col_i+0.5, y_positions[3]+cell_h/2, f'{row["N"]:,}',
            ha='center', va='center', fontsize=9, color='#555555')

for col_i, (_, row) in enumerate(hm.iterrows()):
    is_main = 'Main' in row['Model']
    ax.text(col_i+0.5, y_positions[0]+cell_h+0.25, row['Model'],
            ha='center', va='bottom', fontsize=8.5,
            fontweight='bold' if is_main else 'normal',
            color='#1a1a1a' if is_main else '#555555')

for yp, lbl in zip(y_positions,
                   ['Ford\nCoef', 'P-value', 'Fatal\nCases', 'Total\nObs (N)']):
    ax.text(-0.1, yp+cell_h/2, lbl,
            ha='right', va='center', fontsize=9.5,
            fontweight='bold', color='#1a1a1a')

for yp in y_positions:
    ax.plot([1, 1], [yp, yp+cell_h], color='#555555', lw=2, zorder=3)

ax.text(0.5, y_positions[0]+cell_h+0.65, 'Main Model',
        ha='center', fontsize=9, color='#2c7bb6', fontweight='bold')
ax.text(4.5, y_positions[0]+cell_h+0.65, 'Sub-Sample Models',
        ha='center', fontsize=9, color='#555555')

ax.set_xlim(-1.2, n_models)
ax.set_ylim(-0.3, y_positions[0]+cell_h+1.0)
ax.axis('off')
ax.set_title('Ford Effect Summary Across All Models',
             fontsize=13, fontweight='bold', color='#1a1a1a', pad=10)
fig.text(0.5, -0.02,
         'Coef: red = stronger Ford effect  |  P-value: green = more significant  |  * p < 0.05',
         ha='center', color='#888888', fontsize=9)

plt.tight_layout()
plt.savefig('ford_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()


file.close()
## Close the text file.
