import pandas
import numpy as np



#Use this function for simple transformations of the data like compute score_delta, hour of day scraped, etc
def create_features(_df):
    df = _df.copy()
    df['time_delta'] = df.groupby('id')['age_in_hours'].diff(periods=-1)*-1
    df['score_delta'] = df.groupby('id')['score'].diff(periods=-1)*-1
    df['comment_delta'] = df.groupby('id')['numComments'].diff(periods=-1)*-1

    df['log_score'] = np.log10(np.maximum(1, df['score']))

    df['timeScraped'] = pandas.to_datetime(df['timeScraped'])
    df['dateScraped'] = df['timeScraped'].apply(lambda x: x.date())
    df['day_of_week'] = df['timeScraped'].apply(lambda x: x.weekday())
    df['hour_scraped_est'] = df['timeScraped'].apply(lambda x: x.hour)

    return df



# Use this function to remove invalid data (like things with negative score_delta, etc)

def remove_invalid_observations(_df):
    df = _df[_df.score_delta >= 0].copy()
    return df

def remove_articles_low_observations(_df, MIN_OBSERVATIONS=10):
    return _df.groupby('id').filter(lambda x: x.size >= MIN_OBSERVATIONS)


def remove_observations_time_of_day(_df, min_hour, max_hour):
    return _df[_df['hour_scraped_est'].apply(lambda x: x >= min_hour and x <= max_hour)]