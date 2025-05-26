import pandas as pd
import matplotlib.pyplot as plt
import argparse  # 添加到文件开头的导入部分
import numpy as np
import math
import os
# 1. Read and preprocess data


def calc_mom_change(df):
    """
    Calculate month-over-month change percentage for each URL separately

    Args:
        df: DataFrame with columns ['site_url', 'month_start', 'total_clicks', 'total_impressions', 'avg_rankning']
    Returns:
        DataFrame with added MoM change columns
    """
    # Sort by site_url and month_start
    df = df.sort_values(['site_url', 'month_start'])

    # Calculate MoM changes for each URL separately
    result = df.copy()

    for url in df['site_url'].unique():
        url_mask = df['site_url'] == url
        url_data = df[url_mask].copy()

        # Calculate changes for this URL
        for col in ['total_clicks', 'total_impressions', 'avg_rankning']:
            mom_col = f'{col}_mom'
            # Calculate percentage change
            changes = url_data[col].pct_change(fill_method=None)

            # Handle special cases
            for i in range(1, len(url_data)):
                prev_val = url_data[col].iloc[i-1]
                curr_val = url_data[col].iloc[i]

                if prev_val == 0:
                    if curr_val > 0:
                        changes.iloc[i] = float('inf')  # new
                    elif curr_val == 0:
                        changes.iloc[i] = 0  # no change
                    else:
                        # should not happen for clicks/impressions
                        changes.iloc[i] = float('-inf')

            # Store the changes
            result.loc[url_mask, mom_col] = changes * 100

    return result


def load_and_prepare_data():
    """
    Load and prepare data for analysis
    """
    try:
        print("Loading data...")
        df = pd.read_csv(os.path.join('data', 'keywords_processed.csv') )

        # Convert month_start to datetime
        df['month_start'] = pd.to_datetime(df['month_start'])

        # Convert numeric columns to appropriate types
        numeric_columns = ['total_clicks', 'total_impressions', 'avg_rankning']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add time series features
        df = add_time_series_features(df)

        # Print data validation report
        print("\nData Validation Report:")
        print("-" * 40)
        print(f"Original data rows: {len(df)}")
        print(f"Unique keywords: {df['query'].nunique()}")
        print(
            f"Time range: {df['month_start'].min().strftime('%Y-%m-%d')} to {df['month_start'].max().strftime('%Y-%m-%d')}")
        print(f"Successfully loaded data, {len(df)} records")

        return df

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# 2. Get time series for a single keyword


def get_keyword_time_series(df, keyword):
    """
    Get time series data for specified keyword using fuzzy search, grouped by site_url
    """
    # Use fuzzy search to find matching keywords
    keyword_mask = df['query'].str.contains(keyword, case=False, na=False)
    keyword_data = df[keyword_mask].sort_values(['site_url', 'month_start'])

    if keyword_data.empty:
        print(f"\nNo exact or fuzzy matches found for keyword: {keyword}")
        return keyword_data

  

    for url in keyword_data['site_url'].unique():
        url_data = keyword_data[keyword_data['site_url'] == url]
    

        # Calculate changes between first and last month
        if len(url_data) > 1:
            first_month = url_data.iloc[0]
            last_month = url_data.iloc[-1]

            # print("\n  Changes (last month vs first month):")
            clicks_diff = last_month['total_clicks'] - \
                first_month['total_clicks']
            impressions_diff = last_month['total_impressions'] - \
                first_month['total_impressions']
            ranking_diff = first_month['avg_rankning'] - \
                last_month['avg_rankning']
            # print(f"    Clicks: {clicks_diff:+d}")
            # print(f"    Impressions: {impressions_diff:+d}")
            # print(f"    Avg Rank: {ranking_diff:+.2f}")

    return keyword_data

# 3. Get all keywords' time series as a dict


def get_all_keywords_time_series(df):
    """
    Return a dict: {query: time series DataFrame}
    """
    return {q: subdf.sort_values('month_start') for q, subdf in df.groupby('query')}

# 4. Add last 4 months' clicks, trend description, and MoM changes


def get_last_n_clicks(df, row, n=4):
    q = row['query']
    m = row['month_start']
    sub = df[(df['query'] == q) & (df['month_start'] <= m)
             ].sort_values('month_start').tail(n)
    return list(sub['total_clicks'])


def describe_trend(sequence):
    """
    Describe the trend of a sequence of numbers

    Args:
        sequence: List of numbers to analyze
    """
    if not sequence or len(sequence) < 2:
        return "insufficient_data"

    # Calculate changes
    changes = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]

    # Count positive and negative changes
    positive_changes = sum(1 for x in changes if x > 0)
    negative_changes = sum(1 for x in changes if x < 0)

    # Determine trend
    if positive_changes > negative_changes:
        if positive_changes == len(changes):
            return "continuous_growth"
        return "overall_growth"
    elif negative_changes > positive_changes:
        if negative_changes == len(changes):
            return "continuous_decline"
        return "overall_decline"
    else:
        return "fluctuating"


def format_mom_change(value):
    """
    Format MoM change value for display
    """
    if pd.isna(value):
        return 'first_month'
    elif value == float('inf'):
        return 'new'
    elif value == float('-inf'):
        return 'discontinued'
    else:
        return f"{value:.2f}%"


def calculate_keyword_value_score(row: pd.Series) -> float:
    """
    计算关键词的价值评分

    Args:
        row: 包含关键词数据的Series

    Returns:
        float: 价值评分（0-100）
    """
    try:
        # 基础指标
        ranking = float(row['avg_rankning'])
        clicks = float(row['total_clicks'])
        impressions = float(row['total_impressions'])

        # 计算CTR
        ctr = (clicks / impressions * 100) if impressions > 0 else 0

        # 排名得分 (使用指数衰减，排名越低得分越高)
        ranking_score = 100 * math.exp(-0.1 * ranking)

        # 点击量得分 (使用对数函数，避免点击量差异过大)
        clicks_score = 0
        if clicks > 0:
            clicks_score = 20 * math.log1p(clicks)

        # CTR得分 (线性映射到0-30分)
        ctr_score = min(30, ctr * 3)

        # 计算总分
        total_score = ranking_score + clicks_score + ctr_score

        # 确保分数在0-100之间
        final_score = max(0, min(100, total_score))

        return final_score

    except Exception as e:

        return 0


def add_time_series_features(df):
    """
    添加时序特征，包括价值评分
    """
    # 保持原有的时序特征计算
    df = df.sort_values(['site_url', 'month_start'])

    # 计算MoM变化
    for col in ['total_clicks', 'total_impressions', 'avg_rankning']:
        mom_col = f'{col}_mom'
        if col == 'avg_rankning':
            df[mom_col] = df.groupby('site_url')[col].transform(
                lambda x: ((x.shift(1) - x) / x.shift(1) * 100)
            )
        else:
            df[mom_col] = df.groupby('site_url')[col].pct_change(
                fill_method=None) * 100

    # 添加价值评分
    df['value_score'] = df.apply(calculate_keyword_value_score, axis=1)

    # 计算价值评分的变化
    df['value_score_mom'] = df.groupby('site_url')['value_score'].transform(
        lambda x: x.pct_change(fill_method=None) * 100
    )

    return df

# 5. Visualize time series for a single keyword


def plot_keyword_time_series(df, keyword):
    ts = get_keyword_time_series(df, keyword)
    plt.plot(ts['month_start'], ts['total_clicks'], marker='o')
    plt.title(f"{keyword} - Monthly Clicks Time Series")
    plt.xlabel("Month")
    plt.ylabel("Clicks")
    plt.grid(True)
    plt.show()


def collect_keyword_time_features(df, keyword, n_months=4):
    """
    Collect time series features for keyword
    """
    # Get time series data
    keyword_data = get_keyword_time_series(df, keyword)

    if keyword_data.empty:
        return {
            'keyword': keyword,
            'status': 'no_data',
            'message': f'No data for keyword {keyword}'
        }

    # Get recent n months data
    recent_data = keyword_data.tail(n_months)

    # Collect time features
    time_features = {
        'keyword': keyword,
        'status': 'success',
        'time_range': {
            'start': keyword_data['month_start'].min().strftime('%Y-%m'),
            'end': keyword_data['month_start'].max().strftime('%Y-%m')
        }
    }

    return time_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze keyword time series features')
    parser.add_argument('--keyword', type=str,
                        help='Keyword to analyze (supports fuzzy search)')
    parser.add_argument('--n_months', type=int, default=4,
                        help='Number of recent months to analyze (default: 4)')
    parser.add_argument('--show_all', action='store_true',
                        help='Show time series features for all keywords')
    args = parser.parse_args()

    try:
        # Load and prepare data
        df = pd.read_csv(os.path.join('data', 'keywords_processed.csv') )

        # Convert month_start to datetime
        df['month_start'] = pd.to_datetime(df['month_start'])

        # Convert numeric columns to appropriate types
        numeric_columns = ['total_clicks', 'total_impressions', 'avg_rankning']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Print data validation report
        print("\nData Validation Report:")
        print("-" * 40)
        print(f"Total data rows: {len(df)}")
        print(
            f"Time range: {df['month_start'].min().strftime('%Y-%m-%d')} to {df['month_start'].max().strftime('%Y-%m-%d')}")

        if args.keyword:
            # Analyze specified keyword with fuzzy search
            print(f"\nAnalyzing keyword: {args.keyword}")
            get_keyword_time_series(df, args.keyword)
        elif args.show_all:
            # Analyze all keywords
            print("\nAnalyzing all keywords...")
            all_keywords = df['query'].unique()
            for keyword in all_keywords:
                get_keyword_time_series(df, keyword)
        else:
            # Analyze first three keywords by default
            print("\nAnalyzing first three keywords...")
            sample_keywords = df['query'].unique()[:3]
            for keyword in sample_keywords:
                get_keyword_time_series(df, keyword)

    except Exception as e:
        print(f"Error: {str(e)}")
