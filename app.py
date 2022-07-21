import csv
import datarobot as dr
import pandas as pd
import streamlit as st


def main():
    st.header("GAM Rating Table to Scorecard")
    st.markdown(
        """ This Demo app can be used to convert any DataRobot Rating Table GAM model to Score card with preset scores.
"""
    )
    st.header("Enter Credential Details")
    api_url = st.text_input("DataRobot API URL", value="https://app.datarobot.com/api/v2", max_chars=None, key="api_url", type="default", help=None, autocomplete=None,
                  on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    api_key = st.text_input("DataRobot API Key", value="****", max_chars=None, key="api_key",
                  type="default", help=None, autocomplete=None,
                  on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    #st.button("Connect to DataRobot", key="Connect to DR", help=None, on_click=None, args=None, kwargs=None, disabled=False)
    if st.button('Connect to DR'):
        dr.Client(endpoint=api_url, token=api_key)

    pid = st.text_input("DataRobot Project ID", value="", max_chars=None,
                            key="PID", type="default", help=None, autocomplete=None,
                            on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    mid = st.text_input("DataRobot Model ID", value="", max_chars=None, key="MID",
                            type="default", help=None, autocomplete=None,
                            on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    min = st.number_input("Min Score of Credit Scorecard", value=300, key="min_score")
    max = st.number_input("Min Score of Credit Scorecard", value=900, key="max_score")


    if st.button('Generate Rating Table from Above Model'):
        intercept_score, scorecard = get_scorecard(pid, mid, min_score=min, max_score=max)
        print(st.text('Intercept Score: '))
        print(st.text(intercept_score))
        print(st.text('Rating Table: '))
        print(st.dataframe(scorecard.head()))
        st.download_button(
            label="Download Full Scorecard as CSV",
            data=convert_df_to_csv(scorecard),
            file_name='large_df.csv',
            mime='text/csv',
        )





def download_rating_table(pid, mid):
    """ Download the rating table corresponding to the pid and mid
    """
    project = dr.Project.get(pid)
    rating_tables = rating_tables = project.get_rating_tables()
    #     rating_table_model = dr.RatingTableModel.get(project_id=pid, model_id=mid) # does not work with frozen models
    # Then retrieve the rating table from the model
    #     rating_table_id = rating_table_model.rating_table_id
    #     rating_table = dr.RatingTable.get(pid, rating_table_id)
    rating_table = [rt for rt in rating_tables if rt.model_id == mid][0]
    filepath = './my_rating_table_' + mid + '.csv'
    rating_table.download('./my_rating_table_' + mid + '.csv')
    return filepath


def csv_after_emptylines(filepath, bl_group_n=1, dtype=str):
    """ Read a .CSV into a Pandas DataFrame, but only after at least one blank line has been skipped.
    bl_group_n is the expected number of distinct blocks of blank lines (of any number of rows each) to skip before reading data.
    NB: E.g. pd.read_csv(filepath, skiprows=[0, 1, 2]) works if you know the number of rows to be skipped. Use this function if you have a variable / unknown number of filled rows (to be skipped / ignored) before the empty rows.
    """
    with open(filepath, newline='') as f:
        blank_lines = 0
        bl_groups = 0
        contents = []
        headers = None
        r = csv.reader(f)
        for i, l in enumerate(r):
            if bl_groups < bl_group_n:
                if not l:
                    blank_lines += 1
                    continue
                if blank_lines == 0:
                    continue
                bl_groups += 1
                blank_lines = 0
                headers = l
                continue
            contents.append(l)
        return pd.DataFrame(data=contents, columns=headers, dtype=dtype)


def csv_until_emptyline(filepath, dtype=str):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    with open(filepath, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, l in enumerate(r):
            if not l:
                break
            if i == 0:
                headers = l
                continue
            contents.append(l)
        return pd.DataFrame(data=contents)


def extract_intercept(filepath):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
        Extract intercept value and return it
    """
    df = csv_until_emptyline(filepath)
    df.rename(columns={df.columns[0]: "raw"}, inplace=True)
    df[['name', 'value']] = df['raw'].str.split(":", expand=True)
    intercept = pd.to_numeric(df.loc[df.name == 'Intercept', 'value'].values[0])
    return intercept


def invert_coefficients(intercept, rating_table):
    """ Inverting the sign of intercept and all the coefficients - this is to ensure that the high risk people are given low scores
        Mathematically, we are modelling log of odds and the riskier profiles have high probability
        When we negate the coefficients, it will mean the log of odds of non-risky profiles (- log(p/1-p) = log(1-p/p))
    """
    intercept = - intercept
    rating_table.loc[:, 'Coefficient'] = - rating_table['Coefficient'].astype(float)
    return intercept, rating_table


def convert_rating_table_to_scores(intercept, rating_table, min_score=300, max_score=850):
    rating_table['Rel_Coefficient'] = rating_table['Coefficient']
    baseline = intercept
    min_sum_coef = 0
    max_sum_coef = 0
    for feat in rating_table['Feature Name'].unique():
        min_feat_coef = rating_table.loc[rating_table['Feature Name'] == feat]['Coefficient'].min()
        print('Minimum coefficient for feature ' + feat + ' ' + str(min_feat_coef))
        rating_table.loc[rating_table['Feature Name'] == feat, 'Rel_Coefficient'] = rating_table[
                                                                                        'Coefficient'] - min_feat_coef
        baseline += min_feat_coef
        min_sum_coef = min_sum_coef + rating_table.loc[rating_table['Feature Name'] == feat]['Rel_Coefficient'].min()
        max_sum_coef = max_sum_coef + rating_table.loc[rating_table['Feature Name'] == feat]['Rel_Coefficient'].max()

    min_sum_coef = min_sum_coef + baseline
    max_sum_coef = max_sum_coef + baseline

    rating_table.loc[:, 'Variable Score'] = rating_table['Rel_Coefficient'] * (
                (max_score - min_score) / (max_sum_coef - min_sum_coef))
    baseline_score = (((baseline - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score)) + min_score

    return baseline_score, rating_table.drop(columns=['Coefficient', 'Rel_Coefficient'])


def get_scorecard(pid, mid, min_score=300, max_score=850):
    """ Download rating table for a particular pid and mid and return scorecard
    """
    filepath = download_rating_table(pid, mid)
    rating_table_raw = csv_after_emptylines(filepath)
    intercept_raw = extract_intercept(filepath)
    intercept, rating_table = invert_coefficients(intercept_raw, rating_table_raw)
    intercept_score, scorecard = convert_rating_table_to_scores(intercept, rating_table, min_score, max_score)

    return intercept_score, scorecard

@st.cache
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv().encode('utf-8')



if __name__ == "__main__":
    main()