# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
#     "seaborn",
#     "tabulate",
# ]
# ///
import sys
import os
from typing import Optional
import pandas as pd

# Inline dependency management
# try:
#     import pandas as pd
# except ImportError:
#     import subprocess
#     import sys
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
#     import pandas as pd

def validate_csv(filename: str) -> Optional[pd.DataFrame]:
    """
    Validate and read the CSV file.
    
    Args:
        filename (str): Path to the CSV file
    
    Returns:
        Optional[pd.DataFrame]: Validated DataFrame or None
    """
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return None
        
        # Read CSV file
        df = pd.read_csv(filename,encoding='unicode_escape')
        
        # Basic validation checks
        if df.empty:
            print("Error: CSV file is empty.")
            return None
        
        return df
    
    except pd.errors.EmptyDataError:
        print("Error: No columns to parse from file.")
        return None
    except pd.errors.ParserError:
        print("Error: Unable to parse the CSV file. Please check the file format.")
        return None
    except Exception as e:
        print(f"Unexpected error reading CSV: {e}")
        return None

def generate_readme(df: pd.DataFrame) -> None:
    """
    Generate a README.md file with basic dataset information.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    try:
        with open('README.md', 'w') as f:
            # Basic dataset metadata
            f.write(f"# Dataset Analysis\n\n")
            f.write(f"## Dataset Overview\n")
            f.write(f"- **Total Rows**: {len(df)}\n")
            f.write(f"- **Total Columns**: {len(df.columns)}\n")
            f.write(f"- **Columns**: {', '.join(df.columns)}\n\n")
            
            # Column details
            f.write(f"## Column Details\n")
            for col in df.columns:
                f.write(f"### {col}\n")
                f.write(f"- **Type**: {df[col].dtype}\n")
                f.write(f"- **Non-Null Count**: {df[col].count()}\n")
                f.write(f"- **Unique Values**: {df[col].nunique()}\n\n")


            # # 1. Summary Statistics
            # summary_stats = data.describe().T
            # readme_content.append("## Summary Statistics\n")
            # readme_content.append(summary_stats.to_markdown() + "\n")
    
    except IOError as e:
        print(f"Error writing README.md: {e}")

# def perform_data_analysis(df: pd.DataFrame) -> None:
#     """
#     Perform comprehensive data analysis and generate visualizations.
    
#     Args:
#         df (pd.DataFrame): Input DataFrame
#     """
#     # Summary statistics
#     summary_stats = df.describe()
    
#     # Missing value analysis
#     missing_values = df.isnull().sum()
    
#     # Correlation matrix
#     correlation_matrix = df.corr()
    
#     # Outlier detection using Z-score
#     z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
#     outliers = np.where(z_scores > 3)
    
#     # Clustering (if numerical data available)
#     numeric_df = df.select_dtypes(include=[np.number])
#     if not numeric_df.empty:
#         scaler = StandardScaler()
#         scaled_data = scaler.fit_transform(numeric_df)
        
#         # K-means clustering
#         kmeans = KMeans(n_clusters=3, random_state=42)
#         df['cluster'] = kmeans.fit_predict(scaled_data)


def perform_generic_analysis(df: pd.DataFrame) -> None:
    """
    Perform generic, dataset-independent analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    # Summary statistics for numerical columns
    summary_stats = df.describe()
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Correlation matrix for numerical columns
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    
    # Outlier detection using IQR method
    outliers = {}
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = {
            'lower_outliers': df[df[col] < lower_bound][col].count(),
            'upper_outliers': df[df[col] > upper_bound][col].count()
        }

    with open('README.md', 'w') as f:
        f.write("# Generic Dataset Analysis\n\n")
        f.write("## Summary Statistics\n")
        f.write(summary_stats.to_markdown())
        
        f.write("\n\n## Missing Values\n")
        f.write(missing_values.to_markdown())
        
        f.write("\n\n## Outliers\n")
        for col, stats in outliers.items():
            f.write(f"### {col}\n")
            f.write(f"- Lower outliers: {stats['lower_outliers']}\n")
            f.write(f"- Upper outliers: {stats['upper_outliers']}\n")

def main():
    """
    Main script execution function.
    Validates CSV and generates README.
    """
    # Ensure at least one argument is provided
    if len(sys.argv) < 2:
        print("Usage: python xyz.py <csv_file>")
        sys.exit(1)
    
    # Get CSV filename from command-line arguments
    csv_file = os.path.abspath(sys.argv[1])

    dir_name = os.path.splitext(os.path.basename(csv_file))[0]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_dir_path = os.path.join(script_dir, dir_name)
    os.makedirs(full_dir_path, exist_ok=True)
    
    # Validate and read CSV
    df = validate_csv(csv_file)
    
    if df is not None:

        
        os.chdir(full_dir_path)
        # Generate README
        generate_readme(df)
        # perform_data_analysis(df)
        perform_generic_analysis(df)
        print(f"Analysis complete. README.md generated for {csv_file}")
        # finally:
        #     os.chdir(original_dir)
    else:
        sys.exit(1)
    
    # aiproxy_key = os.environ.get('AIPROXY_KEY')
    # print(aiproxy_key)
    # print(dict(os.environ))
    # print(os.environ.get('AIPROXY_KEY'))

if __name__ == "__main__":
    main()
