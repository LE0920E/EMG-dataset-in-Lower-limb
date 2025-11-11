"""
EMG Data Visualization Tool
===========================

This script provides comprehensive visualization and analysis of EMG data from the Lower Limb dataset.
It includes statistical analysis, signal visualization, and comparative analysis between normal and abnormal groups.

Features:
- Data overview and statistics
- Multi-channel signal visualization  
- Data distribution analysis
- Comparative analysis between groups
- Interactive data exploration

Author: EMG Analysis Tool
Date: 2024
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
warnings.filterwarnings('ignore')

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')  # Fallback for older versions
sns.set_palette("husl")

class EMGDataVisualizer:
    """EMG Data Visualization and Analysis Class"""
    
    def __init__(self, data_path):
        """
        Initialize the visualizer with data path
        
        Args:
            data_path (str): Path to the gaits_data directory
        """
        self.data_path = Path(data_path)
        self.abnormal_path = self.data_path / "Abnormal"
        self.normal_path = self.data_path / "normal"
        
        # Channel information
        self.channels = {
            'Recto Femoral': 'Ch1 - 股直肌 (RF)',
            'Biceps Femoral': 'Ch2 - 股二头肌 (BF)', 
            'Vasto Medial': 'Ch3 - 股内侧肌 (VM)',
            'EMG Semitendinoso': 'Ch4 - 半腱肌 (ST)',
            'Flexo-Extension': 'Ch5 - 膝关节角度'
        }
        
        # Action types
        self.actions = {
            'gait': '步行 (Gait)',
            'sitting': '坐姿伸腿 (Sitting Extension)', 
            'standing': '站立屈膝 (Standing Flexion)'
        }
        
        self.data_cache = {}
        
        # Create output directory for saving plots
        self.output_dir = self.data_path / "plots"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_all_files(self):
        """Load all CSV files from both abnormal and normal directories"""
        print("Loading EMG data files...")
        
        files_info = []
        
        # Load abnormal files
        for file_path in self.abnormal_path.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                file_info = self._extract_file_info(file_path, df, 'abnormal')
                files_info.append(file_info)
                self.data_cache[file_path.name] = df
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Load normal files  
        for file_path in self.normal_path.glob("*.csv"):
            try:
                df = pd.read_csv(file_path)
                file_info = self._extract_file_info(file_path, df, 'normal')
                files_info.append(file_info)
                self.data_cache[file_path.name] = df
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.files_df = pd.DataFrame(files_info)
        print(f"Loaded {len(files_info)} files successfully")
        return self.files_df
    
    def _extract_file_info(self, file_path, df, group):
        """Extract metadata from filename and DataFrame"""
        filename = file_path.stem
        
        # Extract subject ID and action type
        parts = filename.split(' - ')[0].split('A' if group == 'abnormal' else 'N')
        subject_id = parts[0]
        action_code = parts[1].lower()
        
        action_type = None
        for code, full_name in self.actions.items():
            if code in action_code:
                action_type = full_name
                break
        
        return {
            'filename': file_path.name,
            'filepath': str(file_path),
            'group': group,
            'subject_id': subject_id,
            'action_type': action_type,
            'action_code': action_code,
            'num_samples': len(df),
            'duration_seconds': len(df) / 1000,  # 1000 Hz sampling rate
            'ch1_mean': df.iloc[:, 0].mean(),
            'ch1_std': df.iloc[:, 0].std(),
            'ch2_mean': df.iloc[:, 1].mean(),
            'ch2_std': df.iloc[:, 1].std(),
            'ch3_mean': df.iloc[:, 2].mean(),
            'ch3_std': df.iloc[:, 2].std(),
            'ch4_mean': df.iloc[:, 3].mean(),
            'ch4_std': df.iloc[:, 3].std(),
            'ch5_mean': df.iloc[:, 4].mean(),
            'ch5_std': df.iloc[:, 4].std()
        }
    
    def show_data_overview(self):
        """Display comprehensive data overview"""
        if not hasattr(self, 'files_df'):
            self.load_all_files()
        
        print("=" * 80)
        print("EMG DATA OVERVIEW")
        print("=" * 80)
        
        # Basic statistics
        total_files = len(self.files_df)
        abnormal_files = len(self.files_df[self.files_df['group'] == 'abnormal'])
        normal_files = len(self.files_df[self.files_df['group'] == 'normal'])
        
        print(f"Total files: {total_files}")
        print(f"Abnormal group: {abnormal_files} files")
        print(f"Normal group: {normal_files} files")
        print()
        
        # Files by action type
        print("Files by Action Type:")
        action_counts = self.files_df['action_type'].value_counts()
        for action, count in action_counts.items():
            print(f"  {action}: {count} files")
        print()
        
        # Data duration statistics
        print("Data Duration Statistics:")
        duration_stats = self.files_df['duration_seconds'].describe()
        print(f"  Average duration: {duration_stats['mean']:.2f} seconds")
        print(f"  Minimum duration: {duration_stats['min']:.2f} seconds")
        print(f"  Maximum duration: {duration_stats['max']:.2f} seconds")
        print()
        
        # Channel statistics
        print("Channel Signal Statistics (Mean ± Std):")
        for i, (channel, description) in enumerate(self.channels.items(), 1):
            mean_col = f'ch{i}_mean'
            std_col = f'ch{i}_std'
            overall_mean = self.files_df[mean_col].mean()
            overall_std = self.files_df[std_col].mean()
            print(f"  {description}: {overall_mean:.4f} ± {overall_std:.4f}")
    
    def plot_channel_comparison(self, filename=None, start_sample=0, end_sample=1000):
        """Plot comparison of all 5 channels for a specific file"""
        if filename is None:
            # Use first file as example
            filename = self.files_df.iloc[0]['filename']
        
        if filename not in self.data_cache:
            print(f"File {filename} not found in cache")
            return
        
        df = self.data_cache[filename]
        
        # Ensure valid sample range
        end_sample = min(end_sample, len(df))
        
        # Create subplots
        fig, axes = plt.subplots(5, 1, figsize=(12, 10))
        fig.suptitle(f'Multi-Channel EMG Signal - {filename}', fontsize=16, fontweight='bold')
        
        time_axis = np.arange(start_sample, end_sample) / 1000  # Convert to seconds
        
        for i, (channel, description) in enumerate(self.channels.items()):
            ax = axes[i]
            signal = df.iloc[start_sample:end_sample, i]
            
            ax.plot(time_axis, signal, linewidth=1)
            ax.set_ylabel(f'{description}\nAmplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = signal.mean()
            std_val = signal.std()
            ax.text(0.02, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}', 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            if i == 4:  # Last subplot
                ax.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"channel_comparison_{filename.replace(' ', '_').replace('.csv', '')}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_group_comparison(self, action_type='gait', channels_to_plot=None):
        """Compare signals between abnormal and normal groups"""
        if channels_to_plot is None:
            channels_to_plot = [0, 1, 2, 3]  # EMG channels only
        
        # Filter files by action type
        action_files = self.files_df[self.files_df['action_code'].str.contains(action_type)]
        
        if len(action_files) == 0:
            print(f"No files found for action type: {action_type}")
            return
        
        # Get one file from each group
        abnormal_file = action_files[action_files['group'] == 'abnormal'].iloc[0]
        normal_file = action_files[action_files['group'] == 'normal'].iloc[0]
        
        abnormal_df = self.data_cache[abnormal_file['filename']]
        normal_df = self.data_cache[normal_file['filename']]
        
        # Create comparison plot
        n_channels = len(channels_to_plot)
        fig, axes = plt.subplots(n_channels, 2, figsize=(15, 3*n_channels))
        
        if n_channels == 1:
            axes = axes.reshape(1, 2)
        
        fig.suptitle(f'Group Comparison - {self.actions[action_type]}', fontsize=16, fontweight='bold')
        
        for i, channel_idx in enumerate(channels_to_plot):
            # Abnormal group
            ax_abnormal = axes[i, 0]
            abnormal_signal = abnormal_df.iloc[:1000, channel_idx]
            time_axis = np.arange(len(abnormal_signal)) / 1000
            
            ax_abnormal.plot(time_axis, abnormal_signal, color='red', linewidth=1, label='Abnormal')
            ax_abnormal.set_title(f'Abnormal Group - {list(self.channels.keys())[channel_idx]}', fontweight='bold')
            ax_abnormal.set_ylabel('Amplitude')
            ax_abnormal.grid(True, alpha=0.3)
            ax_abnormal.legend()
            
            # Normal group
            ax_normal = axes[i, 1]
            normal_signal = normal_df.iloc[:1000, channel_idx]
            time_axis = np.arange(len(normal_signal)) / 1000
            
            ax_normal.plot(time_axis, normal_signal, color='blue', linewidth=1, label='Normal')
            ax_normal.set_title(f'Normal Group - {list(self.channels.keys())[channel_idx]}', fontweight='bold')
            ax_normal.set_ylabel('Amplitude')
            ax_normal.grid(True, alpha=0.3)
            ax_normal.legend()
            
            if i == n_channels - 1:
                ax_abnormal.set_xlabel('Time (seconds)')
                ax_normal.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"group_comparison_{action_type}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_single_sample_timeseries(self, filename=None):
        """
        图1：单样本多通道时序信号图
        展示原始信号质量与同步性
        """
        if filename is None:
            # Use first file if not specified
            filename = self.files_df.iloc[0]['filename']
        
        if filename not in self.data_cache:
            print(f"File {filename} not found in cache. Please load data first.")
            return
        
        df = self.data_cache[filename]
        file_info = self.files_df[self.files_df['filename'] == filename].iloc[0]
        
        # Create figure with subplots for each channel
        n_channels = len(self.channels)
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3*n_channels))
        
        if n_channels == 1:
            axes = [axes]
        
        fig.suptitle(f'Single Sample Multi-Channel Time Series\nFile: {filename} | Subject: {file_info["subject_id"]} | Action: {file_info["action_type"]} | Group: {file_info["group"]}', 
                    fontsize=14, fontweight='bold')
        
        # Use first 2000 samples for better visualization
        sample_size = min(2000, len(df))
        time_axis = np.arange(sample_size) / 1000  # Convert to seconds
        
        for i, (channel, description) in enumerate(self.channels.items()):
            ax = axes[i]
            signal = df.iloc[:sample_size, i]
            
            # Plot signal
            ax.plot(time_axis, signal, linewidth=1, color='blue', alpha=0.8)
            ax.set_ylabel(f'{description}\nAmplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = signal.mean()
            std_val = signal.std()
            ax.text(0.02, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}', 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            if i == n_channels - 1:  # Last subplot
                ax.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"single_sample_timeseries_{filename.replace(' ', '_').replace('.csv', '')}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Single sample time series plot saved to: {save_path}")
        
        plt.show()
    
    def plot_same_subject_multi_action(self, subject_id=None):
        """
        图2：同一受试者多动作对比图
        对比个体在不同动作下的肌肉激活模式
        """
        if subject_id is None:
            # Use first subject if not specified
            subject_id = self.files_df.iloc[0]['subject_id']
        
        # Filter files for the same subject
        subject_files = self.files_df[self.files_df['subject_id'] == subject_id]
        
        if len(subject_files) == 0:
            print(f"No files found for subject: {subject_id}")
            return
        
        # Get files for different actions
        action_files = {}
        for action in ['gait', 'sitting', 'standing']:
            action_files[action] = subject_files[subject_files['action_code'].str.contains(action)]
        
        # Check if we have all three actions
        available_actions = [action for action, files in action_files.items() if len(files) > 0]
        
        if len(available_actions) < 2:
            print(f"Not enough actions available for subject {subject_id}. Available actions: {available_actions}")
            return
        
        # Create figure with subplots for each action
        n_actions = len(available_actions)
        n_channels = 4  # EMG channels only
        fig, axes = plt.subplots(n_channels, n_actions, figsize=(5*n_actions, 3*n_channels))
        
        if n_channels == 1:
            axes = axes.reshape(1, n_actions)
        
        fig.suptitle(f'Same Subject Multi-Action Comparison\nSubject: {subject_id} | Group: {subject_files.iloc[0]["group"]}', 
                    fontsize=14, fontweight='bold')
        
        # Colors for different actions
        colors = ['blue', 'green', 'red']
        
        for j, action in enumerate(available_actions):
            action_file = action_files[action].iloc[0]
            df = self.data_cache[action_file['filename']]
            
            # Use first 2000 samples for better visualization
            sample_size = min(2000, len(df))
            time_axis = np.arange(sample_size) / 1000  # Convert to seconds
            
            for i in range(n_channels):  # EMG channels only
                ax = axes[i, j] if n_channels > 1 else axes[j]
                signal = df.iloc[:sample_size, i]
                
                # Plot signal
                ax.plot(time_axis, signal, linewidth=1, color=colors[j], alpha=0.8)
                
                # Set titles and labels
                if i == 0:
                    ax.set_title(f'{action.capitalize()}', fontweight='bold')
                
                if j == 0:
                    channel_name = list(self.channels.keys())[i]
                    ax.set_ylabel(f'{channel_name}\nAmplitude', fontsize=10)
                
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = signal.mean()
                std_val = signal.std()
                ax.text(0.02, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}', 
                       transform=ax.transAxes, fontsize=7, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                if i == n_channels - 1:  # Last row
                    ax.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"same_subject_multi_action_{subject_id}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Same subject multi-action comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_group_average_gait_pattern(self):
        """
        图3：正常组与异常组步态平均激活模式图
        揭示两组在步行时的肌电差异
        """
        # Filter gait files only
        gait_files = self.files_df[self.files_df['action_code'].str.contains('gait')]
        
        if len(gait_files) == 0:
            print("No gait files found.")
            return
        
        # Separate by group
        normal_gait = gait_files[gait_files['group'] == 'normal']
        abnormal_gait = gait_files[gait_files['group'] == 'abnormal']
        
        if len(normal_gait) == 0 or len(abnormal_gait) == 0:
            print("Need both normal and abnormal groups for comparison.")
            return
        
        # Find common length for alignment
        min_length = min(
            min([len(self.data_cache[file['filename']]) for _, file in normal_gait.iterrows()]),
            min([len(self.data_cache[file['filename']]) for _, file in abnormal_gait.iterrows()])
        )
        
        # Use first 2000 samples or available minimum
        sample_size = min(2000, min_length)
        
        # Calculate average patterns for each group
        n_channels = 4  # EMG channels only
        normal_avg = np.zeros((n_channels, sample_size))
        abnormal_avg = np.zeros((n_channels, sample_size))
        
        # Calculate normal group average
        for _, file_info in normal_gait.iterrows():
            df = self.data_cache[file_info['filename']]
            for i in range(n_channels):
                normal_avg[i] += df.iloc[:sample_size, i].values
        normal_avg /= len(normal_gait)
        
        # Calculate abnormal group average
        for _, file_info in abnormal_gait.iterrows():
            df = self.data_cache[file_info['filename']]
            for i in range(n_channels):
                abnormal_avg[i] += df.iloc[:sample_size, i].values
        abnormal_avg /= len(abnormal_gait)
        
        # Create comparison plot
        time_axis = np.arange(sample_size) / 1000  # Convert to seconds
        
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3*n_channels))
        
        if n_channels == 1:
            axes = [axes]
        
        fig.suptitle('Normal vs Abnormal Group Average Gait Patterns', fontsize=14, fontweight='bold')
        
        for i in range(n_channels):
            ax = axes[i]
            channel_name = list(self.channels.keys())[i]
            
            # Plot both groups
            ax.plot(time_axis, normal_avg[i], linewidth=1.5, color='blue', alpha=0.8, label='Normal Group')
            ax.plot(time_axis, abnormal_avg[i], linewidth=1.5, color='red', alpha=0.8, label='Abnormal Group')
            
            ax.set_ylabel(f'{channel_name}\nAmplitude', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add statistics
            normal_mean = normal_avg[i].mean()
            normal_std = normal_avg[i].std()
            abnormal_mean = abnormal_avg[i].mean()
            abnormal_std = abnormal_avg[i].std()
            
            ax.text(0.02, 0.95, f'Normal: μ={normal_mean:.4f}, σ={normal_std:.4f}\nAbnormal: μ={abnormal_mean:.4f}, σ={abnormal_std:.4f}', 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            if i == n_channels - 1:  # Last subplot
                ax.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"group_average_gait_pattern_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Group average gait pattern plot saved to: {save_path}")
        
        plt.show()
    
    def plot_rectus_femoris_rms_boxplot(self):
        """
        图4：股四头肌RMS值箱线图对比
        统计比较正常与异常组的肌电强度差异
        """
        # Calculate RMS values for Rectus Femoris channel (channel 0)
        rms_values = []
        groups = []
        subjects = []
        actions = []
        
        for _, file_info in self.files_df.iterrows():
            df = self.data_cache[file_info['filename']]
            
            # Calculate RMS for Rectus Femoris channel
            rectus_femoris_signal = df.iloc[:, 0]  # First channel is Rectus Femoral
            rms_value = np.sqrt(np.mean(rectus_femoris_signal ** 2))
            
            rms_values.append(rms_value)
            groups.append(file_info['group'])
            subjects.append(file_info['subject_id'])
            actions.append(file_info['action_type'])
        
        # Create DataFrame for plotting
        rms_df = pd.DataFrame({
            'RMS': rms_values,
            'Group': groups,
            'Subject': subjects,
            'Action': actions
        })
        
        # Create boxplot
        plt.figure(figsize=(10, 6))
        
        # Boxplot by group
        sns.boxplot(data=rms_df, x='Group', y='RMS', palette=['blue', 'red'])
        
        # Add individual data points
        sns.stripplot(data=rms_df, x='Group', y='RMS', color='black', alpha=0.6, size=4)
        
        plt.title('Rectus Femoris RMS Values Comparison by Group', fontsize=14, fontweight='bold')
        plt.ylabel('RMS Value')
        plt.xlabel('Group')
        
        # Add statistical significance annotation if applicable
        from scipy import stats
        
        normal_rms = rms_df[rms_df['Group'] == 'normal']['RMS']
        abnormal_rms = rms_df[rms_df['Group'] == 'abnormal']['RMS']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(normal_rms, abnormal_rms)
        
        # Add p-value annotation
        if p_value < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_value:.3f}'
        
        plt.text(0.5, 0.95, f'T-test: {p_text}', transform=plt.gca().transAxes, 
                fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add mean values
        normal_mean = normal_rms.mean()
        abnormal_mean = abnormal_rms.mean()
        
        plt.text(0.5, 0.85, f'Normal Mean: {normal_mean:.4f}\nAbnormal Mean: {abnormal_mean:.4f}', 
                transform=plt.gca().transAxes, fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"rectus_femoris_rms_boxplot_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rectus Femoris RMS boxplot saved to: {save_path}")
        
        plt.show()
        
        # Also create boxplot by action type
        plt.figure(figsize=(12, 6))
        
        # Boxplot by group and action
        sns.boxplot(data=rms_df, x='Action', y='RMS', hue='Group', palette=['blue', 'red'])
        
        plt.title('Rectus Femoris RMS Values by Action Type and Group', fontsize=14, fontweight='bold')
        plt.ylabel('RMS Value')
        plt.xlabel('Action Type')
        plt.legend(title='Group')
        
        plt.tight_layout()
        
        # Save the second plot
        save_path = self.output_dir / f"rectus_femoris_rms_by_action_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rectus Femoris RMS by action plot saved to: {save_path}")
        
        plt.show()
    
    def plot_multi_subject_rectus_femoris_heatmap(self, group_type='abnormal', action_type='gait'):
        """
        图5：多受试者股直肌激活热力图
        展示群体内个体间激活一致性
        """
        # Filter files by group and action
        filtered_files = self.files_df[
            (self.files_df['group'] == group_type) & 
            (self.files_df['action_code'].str.contains(action_type))
        ]
        
        if len(filtered_files) == 0:
            print(f"No files found for group '{group_type}' and action '{action_type}'")
            return
        
        # Find common length for alignment
        min_length = min([len(self.data_cache[file['filename']]) for _, file in filtered_files.iterrows()])
        sample_size = min(1000, min_length)  # Use first 1000 samples for heatmap
        
        # Create heatmap data matrix
        n_subjects = len(filtered_files)
        heatmap_data = np.zeros((n_subjects, sample_size))
        subject_ids = []
        
        for idx, (_, file_info) in enumerate(filtered_files.iterrows()):
            df = self.data_cache[file_info['filename']]
            # Use Rectus Femoris channel (channel 0)
            rectus_femoris_signal = df.iloc[:sample_size, 0].values
            heatmap_data[idx] = rectus_femoris_signal
            subject_ids.append(file_info['subject_id'])
        
        # Normalize each subject's data for better visualization
        heatmap_data_normalized = (heatmap_data - heatmap_data.mean(axis=1, keepdims=True)) / heatmap_data.std(axis=1, keepdims=True)
        
        # Create heatmap
        plt.figure(figsize=(15, 8))
        
        # Create heatmap with seaborn
        sns.heatmap(heatmap_data_normalized, 
                   cmap='RdBu_r', 
                   center=0,
                   cbar_kws={'label': 'Normalized Amplitude'},
                   xticklabels=50,  # Show every 50th time point
                   yticklabels=subject_ids)
        
        plt.title(f'Multi-Subject Rectus Femoris Activation Heatmap\nGroup: {group_type.capitalize()} | Action: {action_type.capitalize()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Time (samples)')
        plt.ylabel('Subject ID')
        
        # Add time axis labels in seconds
        time_points = np.arange(0, sample_size, 200)  # Every 0.2 seconds
        plt.xticks(time_points, [f"{t/1000:.1f}" for t in time_points])
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"multi_subject_rectus_femoris_heatmap_{group_type}_{action_type}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-subject Rectus Femoris heatmap saved to: {save_path}")
        
        plt.show()
        
        # Also create correlation heatmap between subjects
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation matrix between subjects
        correlation_matrix = np.corrcoef(heatmap_data)
        
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   xticklabels=subject_ids,
                   yticklabels=subject_ids)
        
        plt.title(f'Inter-Subject Correlation Matrix\nRectus Femoris Activation | Group: {group_type.capitalize()} | Action: {action_type.capitalize()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Subject ID')
        plt.ylabel('Subject ID')
        
        plt.tight_layout()
        
        # Save the correlation heatmap
        save_path = self.output_dir / f"inter_subject_correlation_{group_type}_{action_type}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Inter-subject correlation heatmap saved to: {save_path}")
        
        plt.show()
    
    def plot_emg_feature_pca(self):
        """
        图6：EMG特征空间PCA分布图
        探索正常与异常在特征空间中的可分性
        """
        # Extract features from all files
        features_list = []
        labels_list = []
        file_info_list = []
        
        for _, file_info in self.files_df.iterrows():
            df = self.data_cache[file_info['filename']]
            
            # Extract features for each channel
            channel_features = []
            for channel_idx in range(4):  # 4 EMG channels
                signal = df.iloc[:, channel_idx].values
                
                # Calculate RMS (Root Mean Square)
                rms = np.sqrt(np.mean(signal**2))
                
                # Calculate MAV (Mean Absolute Value)
                mav = np.mean(np.abs(signal))
                
                # Calculate VAR (Variance)
                var = np.var(signal)
                
                # Calculate WL (Waveform Length)
                wl = np.sum(np.abs(np.diff(signal)))
                
                # Calculate SSC (Slope Sign Change)
                diff_signal = np.diff(signal)
                ssc = np.sum((diff_signal[:-1] * diff_signal[1:]) < 0)
                
                # Calculate ZC (Zero Crossing)
                zc = np.sum((signal[:-1] * signal[1:]) < 0)
                
                channel_features.extend([rms, mav, var, wl, ssc, zc])
            
            # Add file metadata
            features_list.append(channel_features)
            labels_list.append(file_info['group'])  # 'normal' or 'abnormal'
            file_info_list.append({
                'subject_id': file_info['subject_id'],
                'action': file_info['action_code'],
                'filename': file_info['filename']
            })
        
        # Convert to numpy array
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA plot
        plt.figure(figsize=(12, 8))
        
        # Plot normal group
        normal_mask = y == 'normal'
        plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                   c='blue', alpha=0.7, s=60, label='Normal Group', edgecolors='black')
        
        # Plot abnormal group
        abnormal_mask = y == 'abnormal'
        plt.scatter(X_pca[abnormal_mask, 0], X_pca[abnormal_mask, 1], 
                   c='red', alpha=0.7, s=60, label='Abnormal Group', edgecolors='black')
        
        # Add labels for some points
        for i, (x, y_val) in enumerate(zip(X_pca, y)):
            if i % 3 == 0:  # Label every 3rd point to avoid clutter
                plt.annotate(file_info_list[i]['subject_id'], 
                           (x[0], x[1]), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
        plt.title('EMG Feature Space PCA Distribution\nNormal vs Abnormal Groups', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add decision boundary (linear separator)
        if len(np.unique(y)) == 2:
            # Fit linear SVM to find decision boundary
            clf = LinearSVC()
            clf.fit(X_pca, y)
            
            # Get the separating hyperplane
            w = clf.coef_[0]
            a = -w[0] / w[1]
            xx = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max())
            yy = a * xx - (clf.intercept_[0]) / w[1]
            
            plt.plot(xx, yy, 'k--', alpha=0.8, label='Decision Boundary')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"emg_feature_pca_distribution_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"EMG feature PCA distribution plot saved to: {save_path}")
        
        plt.show()
        
        # Create feature importance plot (loadings)
        plt.figure(figsize=(14, 8))
        
        # Feature names for 4 channels × 6 features
        feature_names = []
        for channel in ['Rectus_Femoris', 'Biceps_Femoris', 'Vasto_Medial', 'Semitendinosus']:
            for feature in ['RMS', 'MAV', 'VAR', 'WL', 'SSC', 'ZC']:
                feature_names.append(f'{channel}_{feature}')
        
        # Plot PC1 loadings
        plt.subplot(1, 2, 1)
        loadings_pc1 = pca.components_[0]
        plt.barh(range(len(loadings_pc1)), loadings_pc1, alpha=0.7)
        plt.yticks(range(len(loadings_pc1)), feature_names, fontsize=8)
        plt.xlabel('PC1 Loading')
        plt.title('PC1 Feature Loadings')
        plt.grid(True, alpha=0.3)
        
        # Plot PC2 loadings
        plt.subplot(1, 2, 2)
        loadings_pc2 = pca.components_[1]
        plt.barh(range(len(loadings_pc2)), loadings_pc2, alpha=0.7)
        plt.yticks(range(len(loadings_pc2)), feature_names, fontsize=8)
        plt.xlabel('PC2 Loading')
        plt.title('PC2 Feature Loadings')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save feature loadings plot
        save_path = self.output_dir / f"pca_feature_loadings_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA feature loadings plot saved to: {save_path}")
        
        plt.show()
        
        # Print PCA statistics
        print(f"\nPCA Analysis Results:")
        print(f"Total variance explained by PC1 and PC2: {pca.explained_variance_ratio_.sum():.2%}")
        print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.2%}")
        print(f"PC2 variance explained: {pca.explained_variance_ratio_[1]:.2%}")
        
        # Calculate group separation
        if len(np.unique(y)) == 2:
            normal_center = X_pca[normal_mask].mean(axis=0)
            abnormal_center = X_pca[abnormal_mask].mean(axis=0)
            separation_distance = np.linalg.norm(normal_center - abnormal_center)
            print(f"Separation distance between group centers: {separation_distance:.4f}")
    
    def plot_statistical_summary(self):
        """Create comprehensive statistical summary plots - each subplot saved separately"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. File distribution by group and action
        plt.figure(figsize=(8, 6))
        group_action_counts = self.files_df.groupby(['group', 'action_type']).size().unstack()
        group_action_counts.plot(kind='bar')
        plt.title('File Distribution by Group and Action', fontweight='bold')
        plt.ylabel('Number of Files')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = self.output_dir / f"file_distribution_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"File distribution plot saved to: {save_path}")
        plt.close()
        
        # 2. Duration distribution
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.files_df, x='action_type', y='duration_seconds', hue='group')
        plt.title('Data Duration Distribution', fontweight='bold')
        plt.ylabel('Duration (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = self.output_dir / f"duration_distribution_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Duration distribution plot saved to: {save_path}")
        plt.close()
        
        # 3. Channel mean values comparison
        plt.figure(figsize=(8, 6))
        channel_means = []
        for i in range(1, 6):
            mean_col = f'ch{i}_mean'
            channel_data = self.files_df.groupby('group')[mean_col].mean()
            channel_means.append(channel_data)
        
        channel_means_df = pd.DataFrame(channel_means, index=list(self.channels.keys()))
        channel_means_df.plot(kind='bar')
        plt.title('Channel Mean Values by Group', fontweight='bold')
        plt.ylabel('Mean Amplitude')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = self.output_dir / f"channel_means_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Channel means plot saved to: {save_path}")
        plt.close()
        
        # 4. Correlation heatmap (EMG channels only)
        plt.figure(figsize=(8, 6))
        sample_files = self.files_df.sample(min(10, len(self.files_df)))
        correlation_data = []
        
        for _, file_info in sample_files.iterrows():
            df = self.data_cache[file_info['filename']]
            # Use first 1000 samples for correlation
            corr_matrix = df.iloc[:1000, :4].corr().values  # EMG channels only
            correlation_data.append(corr_matrix)
        
        avg_correlation = np.mean(correlation_data, axis=0)
        sns.heatmap(avg_correlation, annot=True, cmap='coolwarm', center=0,
                   xticklabels=list(self.channels.keys())[:4],
                   yticklabels=list(self.channels.keys())[:4])
        plt.title('Average Correlation Matrix (EMG Channels)', fontweight='bold')
        plt.tight_layout()
        save_path = self.output_dir / f"correlation_heatmap_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {save_path}")
        plt.close()
        
        # 5. Signal variance by channel
        plt.figure(figsize=(8, 6))
        channel_vars = []
        for i in range(1, 6):
            std_col = f'ch{i}_std'
            var_data = self.files_df.groupby('group')[std_col].var()
            channel_vars.append(var_data)
        
        channel_vars_df = pd.DataFrame(channel_vars, index=list(self.channels.keys()))
        channel_vars_df.plot(kind='bar')
        plt.title('Channel Variance by Group', fontweight='bold')
        plt.ylabel('Variance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = self.output_dir / f"channel_variance_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Channel variance plot saved to: {save_path}")
        plt.close()
        
        # 6. Action type signal characteristics
        plt.figure(figsize=(8, 6))
        action_counts = self.files_df['action_type'].value_counts()
        action_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('File Distribution by Action Type', fontweight='bold')
        plt.ylabel('')
        plt.tight_layout()
        save_path = self.output_dir / f"action_distribution_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action distribution plot saved to: {save_path}")
        plt.close()
        
        print("All statistical summary plots have been saved separately to the plots directory.")
    
    def interactive_data_explorer(self):
        """Interactive data exploration interface"""
        print("=" * 80)
        print("INTERACTIVE DATA EXPLORER")
        print("=" * 80)
        
        while True:
            print("\nAvailable Options:")
            print("1. Show data overview")
            print("2. Plot specific file channels")
            print("3. Compare groups for specific action")
            print("4. Show statistical summary")
            print("5. List available files")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                self.show_data_overview()
            
            elif choice == '2':
                filename = input("Enter filename (or press Enter for first file): ").strip()
                if not filename:
                    filename = None
                self.plot_channel_comparison(filename)
            
            elif choice == '3':
                print("Available actions: gait, sitting, standing")
                action = input("Enter action type: ").strip()
                self.plot_group_comparison(action)
            
            elif choice == '4':
                self.plot_statistical_summary()
            
            elif choice == '5':
                self._list_available_files()
            
            elif choice == '6':
                print("Exiting interactive explorer...")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    def _list_available_files(self):
        """List all available files with metadata"""
        print("\nAvailable Files:")
        print("-" * 60)
        
        for _, file_info in self.files_df.iterrows():
            print(f"File: {file_info['filename']}")
            print(f"  Group: {file_info['group']}")
            print(f"  Subject: {file_info['subject_id']}")
            print(f"  Action: {file_info['action_type']}")
            print(f"  Duration: {file_info['duration_seconds']:.2f} seconds")
            print(f"  Samples: {file_info['num_samples']}")
            print("-" * 60)


def main():
    """Main function to demonstrate the visualization tool"""
    # Initialize visualizer
    data_path = Path(__file__).parent
    visualizer = EMGDataVisualizer(data_path)
    
    # Load data
    visualizer.load_all_files()
    
    # Run interactive explorer
    visualizer.interactive_data_explorer()


if __name__ == "__main__":
    main()