"""
EMG Data Visualization - Example Usage
======================================

This script demonstrates how to use the EMGDataVisualizer class
for various data analysis and visualization tasks.
"""

import os
import sys
from data_visualization import EMGDataVisualizer

def main():
    """Main demonstration function"""
    
    # Initialize the visualizer
    print("Initializing EMG Data Visualizer...")
    data_path = os.path.dirname(__file__)
    visualizer = EMGDataVisualizer(data_path)
    
    # Load all data files
    print("Loading data files...")
    files_df = visualizer.load_all_files()
    
    # Example 1: Show data overview
    print("\n" + "="*60)
    print("EXAMPLE 1: Data Overview")
    print("="*60)
    visualizer.show_data_overview()
    
    # Example 2: Plot specific file channels
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Channel Signal Visualization")
    print("="*60)
    
    # Get first abnormal gait file
    abnormal_gait_files = files_df[(files_df['group'] == 'abnormal') & 
                                  (files_df['action_code'] == 'gait')]
    
    if len(abnormal_gait_files) > 0:
        sample_file = abnormal_gait_files.iloc[0]['filename']
        print(f"Plotting channels for file: {sample_file}")
        visualizer.plot_channel_comparison(sample_file, start_sample=0, end_sample=2000)
    
    # Example 3: Group comparison
    print("\n" + "="*60)
    print("EXAMPLE 3: Group Comparison (Gait Action)")
    print("="*60)
    visualizer.plot_group_comparison(action_type='gait', channels_to_plot=[0, 1])
    
    # Example 4: Statistical summary
    print("\n" + "="*60)
    print("EXAMPLE 4: Statistical Summary")
    print("="*60)
    visualizer.plot_statistical_summary()
    
    # Example 5: Interactive exploration (commented out by default)
    print("\n" + "="*60)
    print("EXAMPLE 5: Interactive Data Explorer")
    print("="*60)
    print("To use interactive explorer, uncomment the line below:")
    print("# visualizer.interactive_data_explorer()")
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


def quick_start():
    """Quick start function for basic usage"""
    
    print("QUICK START GUIDE")
    print("="*40)
    
    # Initialize and load data
    visualizer = EMGDataVisualizer(os.path.dirname(__file__))
    visualizer.load_all_files()
    
    # Quick overview
    visualizer.show_data_overview()
    
    # Quick visualization
    files_df = visualizer.files_df
    if len(files_df) > 0:
        first_file = files_df.iloc[0]['filename']
        print(f"\nPlotting first file: {first_file}")
        visualizer.plot_channel_comparison(first_file, start_sample=0, end_sample=1000)


def analyze_specific_action(action_type='gait'):
    """Analyze specific action type"""
    
    print(f"\nANALYZING {action_type.upper()} ACTION")
    print("="*40)
    
    visualizer = EMGDataVisualizer(os.path.dirname(__file__))
    visualizer.load_all_files()
    
    # Filter files by action type
    action_files = visualizer.files_df[visualizer.files_df['action_code'] == action_type]
    
    print(f"Found {len(action_files)} files for {action_type} action")
    
    # Show group distribution
    group_counts = action_files['group'].value_counts()
    print("\nGroup distribution:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} files")
    
    # Plot comparison
    visualizer.plot_group_comparison(action_type=action_type)


if __name__ == "__main__":
    # Run the main demonstration
    main()
    
    # Uncomment below for quick start
    # quick_start()
    
    # Uncomment below to analyze specific action
    # analyze_specific_action('gait')     # Analyze gait action
    # analyze_specific_action('sitting')   # Analyze sitting action  
    # analyze_specific_action('standing')  # Analyze standing action