#!/usr/bin/python3
import subprocess
import pandas as pd
from datetime import datetime
import pytz


def run_shell_script(script_path, argument):
    """
    Run a shell script at the given path with one argument and return its standard output as a string.
    
    Args:
        script_path (str): Path to the shell script to execute
        argument (str): A single argument to pass to the shell script
        
    Returns:
        str: The standard output from the script execution
        
    Raises:
        FileNotFoundError: If the script doesn't exist
        PermissionError: If the script isn't executable
        subprocess.CalledProcessError: If the script returns a non-zero exit code
    """
    try:
        # Run the shell script with the provided argument and capture output
        result = subprocess.run(
            [script_path, argument], 
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # If you want to see error output when the script fails
        print(f"Error executing script: {e}")
        print(f"Error output: {e.stderr}")
        raise

def get_default_filename():
    """Generate a default filename using current GMT+8 time"""
    # Get current time in GMT+8
    tz = pytz.timezone('Asia/Shanghai')  # Shanghai is in GMT+8
    now = datetime.now(tz)
    # Format as YYYY-MM-DD_HHMMSS
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")
    return f"results_{timestamp}.xlsx"

def process_output_to_excel(output_string, output_file=None):
    """
    Process the string output from a shell script into an Excel file with multiple sheets.
    
    Args:
        output_string (str): The string output from the shell script
        output_file (str, optional): The name of the output Excel file. 
                                     If None, uses current GMT+8 time.
        
    Returns:
        str: Path to the created Excel file
    """
    # Set default filename with GMT+8 timestamp if not provided
    if output_file is None:
        output_file = get_default_filename()
        
    # Split the output into sheets based on blank lines
    sheets_raw = output_string.split('\n\n')
    
    # Create a pandas ExcelWriter object
    with pd.ExcelWriter(output_file) as writer:
        for sheet_raw in sheets_raw:
            lines = sheet_raw.strip().split('\n')
            
            # First line is the sheet name
            sheet_name = lines[0]
            
            # Second line contains column names
            column_names = lines[1].split(',')
            
            # The rest of the lines contain data
            data_values = lines[2:]
            
            # Group data into rows based on the number of columns
            data_rows = []
            row = []
            for value in data_values:
                row.append(float(value))
                if len(row) == len(column_names):
                    data_rows.append(row)
                    row = []
            
            # Create DataFrame and write to Excel
            df = pd.DataFrame(data_rows, columns=column_names)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Run the script and print the output
output = run_shell_script('./eval.sh', '1')
process_output_to_excel(output)
