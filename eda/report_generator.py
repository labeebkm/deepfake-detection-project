"""
Generate HTML/PDF EDA reports.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from jinja2 import Template


class ReportGenerator:
    """Generate comprehensive EDA reports."""
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize ReportGenerator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
    
    def generate_html_report(self, analysis_results: Dict, dataset_name: str, 
                            visualization_paths: Optional[Dict[str, str]] = None) -> str:
        """
        Generate HTML EDA report.
        
        Args:
            analysis_results: Dictionary with all analysis results
            dataset_name: Name of the dataset
            visualization_paths: Optional dictionary mapping visualization names to file paths
            
        Returns:
            Path to generated HTML report
        """
        visualization_paths = visualization_paths or {}
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - {{ dataset_name }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .metric {
            display: inline-block;
            margin: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        .visualization img {
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #667eea;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .recommendation {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .warning {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .success {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Exploratory Data Analysis Report</h1>
        <p><strong>Dataset:</strong> {{ dataset_name }}</p>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
    </div>

    <!-- Dataset Overview -->
    <div class="section">
        <h2>Dataset Overview</h2>
        {% if 'dataset_structure' in analysis_results %}
        <div class="metric">
            <div class="metric-label">Total Images</div>
            <div class="metric-value">{{ analysis_results.dataset_structure.total_images }}</div>
        </div>
        {% endif %}
        {% if 'class_distribution' in analysis_results %}
        <div class="metric">
            <div class="metric-label">Class Imbalance Ratio</div>
            <div class="metric-value">{{ "%.2f"|format(analysis_results.class_distribution.imbalance_ratio) }}</div>
        </div>
        {% endif %}
        {% if 'quality_score' in analysis_results %}
        <div class="metric">
            <div class="metric-label">Quality Score</div>
            <div class="metric-value">{{ "%.2f"|format(analysis_results.quality_score.overall_score) }}</div>
        </div>
        {% endif %}
    </div>

    <!-- Class Distribution -->
    {% if 'class_distribution' in analysis_results %}
    <div class="section">
        <h2>Class Distribution</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {% for class_name, count in analysis_results.class_distribution.counts.items() %}
            <tr>
                <td>{{ class_name }}</td>
                <td>{{ count }}</td>
                <td>{{ "%.2f"|format(analysis_results.class_distribution.percentages[class_name]) }}%</td>
            </tr>
            {% endfor %}
        </table>
        {% if analysis_results.class_distribution.imbalance_ratio > 2 %}
        <div class="warning">
            <strong>Warning:</strong> Significant class imbalance detected. Consider using class weights or data augmentation.
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- Image Statistics -->
    {% if 'image_statistics' in analysis_results %}
    <div class="section">
        <h2>Image Statistics</h2>
        <h3>Resolution Distribution</h3>
        <div class="metric">
            <div class="metric-label">Mean Width</div>
            <div class="metric-value">{{ "%.0f"|format(analysis_results.image_statistics.resolutions.mean) }} px</div>
        </div>
        <div class="metric">
            <div class="metric-label">Mean Height</div>
            <div class="metric-value">{{ "%.0f"|format(analysis_results.image_statistics.resolutions.mean) }} px</div>
        </div>
        {% if 'class_distribution' in visualization_paths %}
        <div class="visualization">
            <img src="{{ visualization_paths.class_distribution }}" alt="Class Distribution">
        </div>
        {% endif %}
    </div>
    {% endif %}

    <!-- Quality Recommendations -->
    {% if 'quality_score' in analysis_results and analysis_results.quality_score.recommendations %}
    <div class="section">
        <h2>Quality Recommendations</h2>
        {% for recommendation in analysis_results.quality_score.recommendations %}
        <div class="recommendation">
            {{ recommendation }}
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Statistical Tests -->
    {% if 'statistical_tests' in analysis_results %}
    <div class="section">
        <h2>Statistical Tests</h2>
        <p>Statistical significance tests comparing real vs fake images.</p>
        <!-- Add statistical test results here -->
    </div>
    {% endif %}

    <!-- Artifact Analysis -->
    {% if 'artifact_analysis' in analysis_results %}
    <div class="section">
        <h2>Artifact Analysis</h2>
        <p>Analysis of deepfake artifacts detected in the dataset.</p>
        <!-- Add artifact analysis results here -->
    </div>
    {% endif %}

    <div class="footer">
        <p>Report generated by Deepfake Detection EDA System</p>
        <p>For questions or issues, please refer to the project documentation.</p>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            dataset_name=dataset_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            analysis_results=analysis_results,
            visualization_paths=visualization_paths
        )
        
        # Save HTML report
        report_path = self.output_dir / f"eda_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {report_path}")
        return str(report_path)
    
    def generate_summary_report(self, analysis_results: Dict, dataset_name: str) -> Dict:
        """
        Generate summary statistics report.
        
        Args:
            analysis_results: Dictionary with all analysis results
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "overview": {},
            "class_distribution": {},
            "image_statistics": {},
            "quality_metrics": {},
            "recommendations": []
        }
        
        # Extract overview
        if "dataset_structure" in analysis_results:
            summary["overview"] = {
                "total_images": analysis_results["dataset_structure"].get("total_images", 0),
                "num_classes": len(analysis_results["dataset_structure"].get("classes", {})),
                "file_formats": dict(analysis_results["dataset_structure"].get("file_formats", {}))
            }
        
        # Extract class distribution
        if "class_distribution" in analysis_results:
            summary["class_distribution"] = {
                "counts": analysis_results["class_distribution"].get("counts", {}),
                "percentages": analysis_results["class_distribution"].get("percentages", {}),
                "imbalance_ratio": analysis_results["class_distribution"].get("imbalance_ratio", 1.0)
            }
        
        # Extract image statistics
        if "image_statistics" in analysis_results:
            img_stats = analysis_results["image_statistics"]
            summary["image_statistics"] = {
                "mean_resolution": img_stats.get("resolutions", {}).get("mean", 0),
                "mean_aspect_ratio": img_stats.get("aspect_ratios", {}).get("mean", 0)
            }
        
        # Extract quality metrics
        if "quality_score" in analysis_results:
            quality = analysis_results["quality_score"]
            summary["quality_metrics"] = {
                "overall_score": quality.get("overall_score", 0),
                "factors": quality.get("factors", {})
            }
            summary["recommendations"] = quality.get("recommendations", [])
        
        # Save summary
        summary_path = self.output_dir / "statistics" / f"summary_{dataset_name}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def generate_comparison_report(self, datasets: List[Dict], dataset_names: List[str]) -> str:
        """
        Generate comparison report for multiple datasets.
        
        Args:
            datasets: List of analysis result dictionaries
            dataset_names: List of dataset names
            
        Returns:
            Path to generated comparison report
        """
        # Create comparison table
        comparison_data = []
        for name, results in zip(dataset_names, datasets):
            row = {"dataset": name}
            
            if "dataset_structure" in results:
                row["total_images"] = results["dataset_structure"].get("total_images", 0)
            
            if "class_distribution" in results:
                row["imbalance_ratio"] = results["class_distribution"].get("imbalance_ratio", 1.0)
            
            if "quality_score" in results:
                row["quality_score"] = results["quality_score"].get("overall_score", 0)
            
            comparison_data.append(row)
        
        # Generate HTML comparison report
        comparison_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #667eea; color: white; }}
    </style>
</head>
<body>
    <h1>Dataset Comparison Report</h1>
    <table>
        <tr>
            <th>Dataset</th>
            <th>Total Images</th>
            <th>Imbalance Ratio</th>
            <th>Quality Score</th>
        </tr>
        {"".join([f"<tr><td>{row['dataset']}</td><td>{row.get('total_images', 'N/A')}</td><td>{row.get('imbalance_ratio', 'N/A'):.2f}</td><td>{row.get('quality_score', 'N/A'):.2f}</td></tr>" for row in comparison_data])}
    </table>
</body>
</html>
        """
        
        report_path = self.output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w') as f:
            f.write(comparison_html)
        
        print(f"Comparison report generated: {report_path}")
        return str(report_path)


def generate_eda_report(dataset_path: str, 
                        output_format: str = 'html',
                        output_dir: str = './reports',
                        dataset_name: Optional[str] = None,
                        config_path: Optional[str] = None) -> str:
    """
    Generate comprehensive EDA report for a dataset.
    
    Args:
        dataset_path: Path to dataset directory
        output_format: Output format ('html' or 'pdf')
        output_dir: Output directory for reports
        dataset_name: Name of the dataset (auto-detected if None)
        config_path: Path to EDA configuration file
        
    Returns:
        Path to generated report
    """
    import json
    from eda.data_analyzer import DataAnalyzer
    from eda.visualization import EDAVisualizer
    
    # Load config if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Auto-detect dataset name from path if not provided
    if dataset_name is None:
        dataset_name = Path(dataset_path).name
    
    # Initialize components
    analyzer = DataAnalyzer(dataset_path, config)
    visualizer = EDAVisualizer(interactive=True)
    report_gen = ReportGenerator(output_dir=output_dir)
    
    # Perform analyses
    print("Analyzing dataset structure...")
    structure = analyzer.analyze_dataset_structure()
    
    print("Analyzing class distribution...")
    class_dist = analyzer.analyze_class_distribution()
    
    print("Analyzing image statistics...")
    img_stats = analyzer.analyze_image_statistics(sample_size=1000)
    
    print("Calculating quality score...")
    quality = analyzer.calculate_dataset_quality_score()
    
    # Generate visualizations
    print("Generating visualizations...")
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    visualization_paths = {}
    
    # Class distribution
    vis_path = os.path.join(output_dir, 'visualizations', 'class_distribution.png')
    visualizer.plot_class_distribution(
        class_dist['counts'],
        save_path=vis_path,
        interactive=False
    )
    visualization_paths['class_distribution'] = vis_path
    
    # Compile results
    analysis_results = {
        'dataset_structure': structure,
        'class_distribution': class_dist,
        'image_statistics': img_stats,
        'quality_score': quality
    }
    
    # Generate report
    if output_format == 'html':
        report_path = report_gen.generate_html_report(
            analysis_results,
            dataset_name,
            visualization_paths
        )
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    print(f"\nEDA report generated: {report_path}")
    return report_path

