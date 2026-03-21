import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_analytics_graphs():
    csv_file = "crowd_metrics.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run process_video.py first.")
        return

    # Read the tracked data
    df = pd.read_csv(csv_file)
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    
    # ==========================================
    # Graph 1: Crowd Density Over Time (Line Chart)
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['Frame Number'], df['Total Detected'], label='Total People', color='blue', linewidth=2)
    plt.plot(df['Frame Number'], df['In Alert Area'], label='People in Alert Area', color='red', linewidth=2)
    
    # Add a dashed line showing the threshold
    threshold = 25 # Matches the threshold set in the script
    plt.axhline(y=threshold, color='orange', linestyle='--', label=f'Alert Threshold ({threshold})')
    
    plt.title('Crowd Density Over Time', fontsize=16)
    plt.xlabel('Video Frame Range', fontsize=12)
    plt.ylabel('Number of People', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    # Save & Show
    plt.savefig('crowd_density_timeline.png', dpi=300)
    plt.show()

    # ==========================================
    # Graph 2: Alert Status Distribution (Pie Chart)
    # ==========================================
    plt.figure(figsize=(8, 8))
    
    alert_counts = df['Alert Triggered'].value_counts()
    
    # Ensure both True and False are represented even if 0
    if True not in alert_counts: alert_counts[True] = 0
    if False not in alert_counts: alert_counts[False] = 0
        
    labels = ['Normal (Under Threshold)', 'ALERT (Over Threshold)']
    sizes = [alert_counts[False], alert_counts[True]]
    colors = ['#2ecc71', '#e74c3c'] # Green and Red
    explode = (0, 0.1)  # slightly "explode" the Alert slice
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=90)
    
    plt.title('Percentage of Time in Alert State', fontsize=16)
    
    # Save & Show
    plt.savefig('alert_distribution.png', dpi=300)
    plt.show()
    
    print("Analytics generation complete! Graphs saved as PNGs.")

if __name__ == "__main__":
    create_analytics_graphs()
