"""
Generate EDA reports from command line.
"""

import argparse
import os
import json
from pathlib import Path

from eda.data_analyzer import DataAnalyzer
from eda.visualization import EDAVisualizer
from eda.statistical_tests import StatisticalTester
from eda.artifact_detector import ArtifactDetector
from eda.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate EDA report for deepfake dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., faceforensics, celebdf)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='./reports',
                       help='Output directory for reports')
    parser.add_argument('--config', type=str, default='configs/eda_config.json',
                       help='Path to EDA configuration file')
    parser.add_argument('--format', type=str, default='html', choices=['html', 'pdf'],
                       help='Report format')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--save', action='store_true',
                       help='Save statistics')
    
    args = parser.parse_args()
    
    # Load EDA config
    with open(args.config, 'r') as f:
        eda_config = json.load(f)
    
    # Initialize components
    analyzer = DataAnalyzer(args.data_dir, eda_config)
    visualizer = EDAVisualizer(interactive=True)
    tester = StatisticalTester(alpha=eda_config.get('eda', {}).get('statistical_tests', {}).get('alpha', 0.05))
    detector = ArtifactDetector()
    report_gen = ReportGenerator(output_dir=args.output)
    
    print(f"Analyzing dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    
    # Perform analyses
    print("\n1. Analyzing dataset structure...")
    structure = analyzer.analyze_dataset_structure()
    
    print("2. Analyzing class distribution...")
    class_dist = analyzer.analyze_class_distribution()
    
    print("3. Analyzing image statistics...")
    img_stats = analyzer.analyze_image_statistics(sample_size=1000)
    
    print("4. Calculating quality score...")
    quality = analyzer.calculate_dataset_quality_score()
    
    # Generate visualizations if requested
    visualization_paths = {}
    if args.visualize:
        print("5. Generating visualizations...")
        os.makedirs(os.path.join(args.output, 'visualizations'), exist_ok=True)
        
        # Class distribution
        vis_path = os.path.join(args.output, 'visualizations', 'class_distribution.png')
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
    
    # Save statistics if requested
    if args.save:
        print("6. Saving statistics...")
        os.makedirs(os.path.join(args.output, 'statistics'), exist_ok=True)
        analyzer.save_statistics(
            os.path.join(args.output, 'statistics', f'dataset_stats_{args.dataset}.json')
        )
    
    # Generate report
    print("7. Generating report...")
    if args.format == 'html':
        report_path = report_gen.generate_html_report(
            analysis_results,
            args.dataset,
            visualization_paths
        )
        print(f"\nEDA report generated: {report_path}")
    else:
        print("PDF generation not yet implemented")
    
    print("\nEDA analysis complete!")


if __name__ == '__main__':
    main()








