import pandas as pd
import matplotlib.pyplot as plt
import ast
import io
try:
    df = pd.read_csv('clip_similarity_results_fma.csv')
except FileNotFoundError:
    print("File not found. Please ensure 'clip_similarity_results_fma.csv' is in the same directory.")
    # Create an empty DataFrame to avoid errors. Please provide the file for actual execution.
    df = pd.DataFrame(columns=['top_3_songs']) 

# 2. Data preprocessing
# Convert string "['song1', 'song2']" to Python List
def parse_list_string(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

if not df.empty:
    df['top_3_songs_list'] = df['top_3_songs'].apply(parse_list_string)

    # 3. Count songs by ranking
    song_stats = {}

    for songs in df['top_3_songs_list']:
        if len(songs) >= 3:
            rank1, rank2, rank3 = songs[0], songs[1], songs[2]
            
            # Initialize
            for s in [rank1, rank2, rank3]:
                if s not in song_stats:
                    song_stats[s] = {'Rank 1': 0, 'Rank 2': 0, 'Rank 3': 0}
            
            # Count
            song_stats[rank1]['Rank 1'] += 1
            song_stats[rank2]['Rank 2'] += 1
            song_stats[rank3]['Rank 3'] += 1

    # Convert to DataFrame
    stats_df = pd.DataFrame.from_dict(song_stats, orient='index')
    
    # Calculate total for sorting
    stats_df['Total'] = stats_df.sum(axis=1)
    
    # Sort and get top 30 (adjustable as needed)
    top_n = 30
    plot_df = stats_df.sort_values(by='Total', ascending=False).head(top_n)
    
    # Remove Total column for plotting
    plot_df = plot_df.drop(columns=['Total'])

    # 4. Plot stacked bar chart
    # Set font for Chinese support (can be omitted or set to sans-serif)
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors: Rank 1 (gold/red), Rank 2 (silver/blue), Rank 3 (bronze/green)
    colors = ['#FF6F61', '#6B5B95', '#88B04B']
    
    plot_df.plot(kind='bar', stacked=False, ax=ax, color=colors, width=0.8)
    
    # plot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)

    # Chart decorations
    plt.title(f'Top {top_n} Most Frequent Songs (Ranked by Top 1~3)', fontsize=16)
    plt.xlabel('Song ID (Filename)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='Ranking', labels=['Rank 1', 'Rank 2', 'Rank 3'])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Display chart
    plt.savefig('retrive_result.png')
else:
    print("DataFrame is empty. Please check the CSV file path.")