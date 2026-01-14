"""
Custom visualization functions for deepfake detection EDA.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Any
import cv2
from pathlib import Path
import pandas as pd
from scipy.fft import dct


class EDAVisualizer:
    """Visualization utilities for EDA analysis."""
    
    def __init__(self, style: str = "seaborn", color_palette: str = "Set2", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            color_palette: Seaborn color palette
            figsize: Default figure size
        """
        if style is not None and style in plt.style.available:
            plt.style.use(style)
        else:
            plt.style.use("default")
        sns.set_palette(color_palette)
        self.figsize = figsize
        self.color_palette = color_palette
        
    def plot_class_distribution(self, class_counts: Dict[str, int], save_path: Optional[str] = None, interactive: bool = True):
        """
        Plot class distribution (real vs fake).
        
        Args:
            class_counts: Dictionary with class counts
            save_path: Optional path to save figure
            interactive: Whether to create interactive plotly plot
        """
        if interactive:
            fig = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="Class Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax.set_title("Class Distribution")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
    
    def plot_resolution_distribution(self, resolutions: List[Tuple[int, int]], save_path: Optional[str] = None, interactive: bool = True):
        """
        Plot image resolution distribution.
        
        Args:
            resolutions: List of (width, height) tuples
            save_path: Optional path to save figure
            interactive: Whether to create interactive plotly plot
        """
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        
        if interactive:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Width Distribution', 'Height Distribution')
            )
            fig.add_trace(go.Histogram(x=widths, name='Width', nbinsx=50), row=1, col=1)
            fig.add_trace(go.Histogram(x=heights, name='Height', nbinsx=50), row=1, col=2)
            fig.update_layout(title_text="Resolution Distribution", showlegend=False)
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            return fig
        else:
            fig, axes = plt.subplots(1, 2, figsize=self.figsize)
            axes[0].hist(widths, bins=50, edgecolor='black')
            axes[0].set_xlabel('Width (pixels)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Width Distribution')
            
            axes[1].hist(heights, bins=50, edgecolor='black')
            axes[1].set_xlabel('Height (pixels)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Height Distribution')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
    
    def plot_side_by_side_comparison(self, real_images: List[np.ndarray], fake_images: List[np.ndarray], 
                                     num_samples: int = 10, save_path: Optional[str] = None):
        """
        Create side-by-side comparison of real vs fake images.
        
        Args:
            real_images: List of real image arrays
            fake_images: List of fake image arrays
            num_samples: Number of samples to display
            save_path: Optional path to save figure
        """
        num_samples = min(num_samples, len(real_images), len(fake_images))
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
        
        for i in range(num_samples):
            # Real image
            if len(real_images[i].shape) == 3:
                axes[i, 0].imshow(real_images[i])
            else:
                axes[i, 0].imshow(real_images[i], cmap='gray')
            axes[i, 0].set_title('Real')
            axes[i, 0].axis('off')
            
            # Fake image
            if len(fake_images[i].shape) == 3:
                axes[i, 1].imshow(fake_images[i])
            else:
                axes[i, 1].imshow(fake_images[i], cmap='gray')
            axes[i, 1].set_title('Fake')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_difference_map(self, real_image: np.ndarray, fake_image: np.ndarray, 
                           method: str = "absolute", save_path: Optional[str] = None):
        """
        Plot difference map between real and fake images.
        
        Args:
            real_image: Real image array
            fake_image: Fake image array
            method: 'absolute', 'squared', or 'diff'
            save_path: Optional path to save figure
        """
        # Ensure same size
        if real_image.shape != fake_image.shape:
            fake_image = cv2.resize(fake_image, (real_image.shape[1], real_image.shape[0]))
        
        if method == "absolute":
            diff = np.abs(real_image.astype(float) - fake_image.astype(float))
        elif method == "squared":
            diff = np.square(real_image.astype(float) - fake_image.astype(float))
        else:
            diff = real_image.astype(float) - fake_image.astype(float)
        
        # Normalize for visualization
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(real_image)
        axes[0].set_title('Real')
        axes[0].axis('off')
        
        axes[1].imshow(fake_image)
        axes[1].set_title('Fake')
        axes[1].axis('off')
        
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f'Difference Map ({method})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_frequency_spectrum(self, real_image: np.ndarray, fake_image: np.ndarray, 
                               save_path: Optional[str] = None):
        """
        Plot frequency spectrum comparison using DCT.
        
        Args:
            real_image: Real image array
            fake_image: Fake image array
            save_path: Optional path to save figure
        """
        # Convert to grayscale if needed
        if len(real_image.shape) == 3:
            real_gray = cv2.cvtColor(real_image, cv2.COLOR_RGB2GRAY)
        else:
            real_gray = real_image
        
        if len(fake_image.shape) == 3:
            fake_gray = cv2.cvtColor(fake_image, cv2.COLOR_RGB2GRAY)
        else:
            fake_gray = fake_image
        
        # Apply DCT transform
        real_freq = np.abs(dct(dct(real_gray, axis=0, norm='ortho'), axis=1, norm='ortho'))
        fake_freq = np.abs(dct(dct(fake_gray, axis=0, norm='ortho'), axis=1, norm='ortho'))
        
        # Log scale for better visualization
        real_freq_log = np.log(real_freq + 1)
        fake_freq_log = np.log(fake_freq + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(real_image)
        axes[0, 0].set_title('Real Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(fake_image)
        axes[0, 1].set_title('Fake Image')
        axes[0, 1].axis('off')
        
        im1 = axes[1, 0].imshow(real_freq_log, cmap='viridis')
        axes[1, 0].set_title('Real DCT Spectrum')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(fake_freq_log, cmap='viridis')
        axes[1, 1].set_title('Fake DCT Spectrum')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_histogram_comparison(self, real_image: np.ndarray, fake_image: np.ndarray, 
                                 channels: List[str] = ['R', 'G', 'B'], save_path: Optional[str] = None, interactive: bool = True):
        """
        Plot RGB histogram comparison.
        
        Args:
            real_image: Real image array
            fake_image: Fake image array
            channels: List of channel names
            save_path: Optional path to save figure
            interactive: Whether to create interactive plotly plot
        """
        if interactive:
            fig = make_subplots(
                rows=1, cols=len(channels),
                subplot_titles=[f'{ch} Channel' for ch in channels]
            )
            
            colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
            for idx, ch in enumerate(channels):
                ch_idx = ['R', 'G', 'B'].index(ch) if ch in ['R', 'G', 'B'] else 0
                
                if len(real_image.shape) == 3:
                    real_hist = np.histogram(real_image[:, :, ch_idx], bins=256, range=(0, 256))[0]
                    fake_hist = np.histogram(fake_image[:, :, ch_idx], bins=256, range=(0, 256))[0]
                else:
                    real_hist = np.histogram(real_image, bins=256, range=(0, 256))[0]
                    fake_hist = np.histogram(fake_image, bins=256, range=(0, 256))[0]
                
                x = np.arange(256)
                fig.add_trace(go.Scatter(x=x, y=real_hist, name='Real', line=dict(color=colors.get(ch, 'black'))), row=1, col=idx+1)
                fig.add_trace(go.Scatter(x=x, y=fake_hist, name='Fake', line=dict(color=colors.get(ch, 'gray'), dash='dash')), row=1, col=idx+1)
            
            fig.update_layout(title_text="RGB Histogram Comparison", showlegend=True)
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            return fig
        else:
            fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 5))
            if len(channels) == 1:
                axes = [axes]
            
            colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
            for idx, ch in enumerate(channels):
                ch_idx = ['R', 'G', 'B'].index(ch) if ch in ['R', 'G', 'B'] else 0
                
                if len(real_image.shape) == 3:
                    axes[idx].hist(real_image[:, :, ch_idx].ravel(), bins=256, alpha=0.5, label='Real', color=colors.get(ch, 'black'))
                    axes[idx].hist(fake_image[:, :, ch_idx].ravel(), bins=256, alpha=0.5, label='Fake', color='gray')
                else:
                    axes[idx].hist(real_image.ravel(), bins=256, alpha=0.5, label='Real', color='black')
                    axes[idx].hist(fake_image.ravel(), bins=256, alpha=0.5, label='Fake', color='gray')
                
                axes[idx].set_xlabel('Pixel Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'{ch} Channel')
                axes[idx].legend()
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
    
    def plot_tsne_visualization(self, features: np.ndarray, labels: np.ndarray, save_path: Optional[str] = None, interactive: bool = True):
        """
        Plot t-SNE visualization of features.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label array (n_samples,)
            save_path: Optional path to save figure
            interactive: Whether to create interactive plotly plot
        """
        from sklearn.manifold import TSNE
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        if interactive:
            df = pd.DataFrame({
                'x': features_2d[:, 0],
                'y': features_2d[:, 1],
                'label': labels
            })
            fig = px.scatter(df, x='x', y='y', color='label', title='t-SNE Visualization')
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], label=f'Class {label}', alpha=0.6)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_title('t-SNE Visualization')
            ax.legend()
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
    
    def plot_correlation_matrix(self, features: pd.DataFrame, save_path: Optional[str] = None, interactive: bool = True):
        """
        Plot correlation matrix of features.
        
        Args:
            features: DataFrame with features
            save_path: Optional path to save figure
            interactive: Whether to create interactive plotly plot
        """
        corr_matrix = features.corr()
        
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(title='Feature Correlation Matrix')
            if save_path:
                fig.write_html(save_path.replace('.png', '.html'))
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
    
    def plot_frequency_comparison(self, real_image: np.ndarray, fake_image: np.ndarray,
                                  save_path: Optional[str] = None):
        """
        Plot frequency comparison between real and fake images using DCT.
        
        This is an alias for plot_frequency_spectrum with a more convenient name.
        
        Args:
            real_image: Real image array
            fake_image: Fake image array
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        return self.plot_frequency_spectrum(real_image, fake_image, save_path=save_path)

