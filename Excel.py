import os

# Replace with your actual file path
file_path = r"C:\Gilo\myrepo\test.xlsx"

try:
    os.startfile(file_path)
    print("Excel file opened successfully.")
except FileNotFoundError:
    print("Error: File not found. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")