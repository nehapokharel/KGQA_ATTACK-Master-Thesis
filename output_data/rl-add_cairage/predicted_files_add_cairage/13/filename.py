import os

def get_filenames_from_folder(folder_path):
    """
    Gets a list of filenames from a specified folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of filenames. Returns an empty list if the
              folder does not exist or is not a directory.
        None: If an error occurs (e.g., permission denied).
    """
    try:
        # Check if the path exists and is a directory
        if not os.path.isdir(folder_path):
            print(f"Error: Folder not found at '{folder_path}'")
            return []

        # Get all entries in the directory and filter for files
        # os.listdir() returns everything (files, folders, links, etc.)
        # os.path.isfile() checks if an entry is a regular file.
        filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        return filenames

    except FileNotFoundError:
        print(f"Error: The directory '{folder_path}' does not exist.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to access '{folder_path}'.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Example Usage ---

# 1. Specify the path to your folder.
#    - Use '.' for the current directory.
#    - Use an absolute path (e.g., 'C:/Users/YourUser/Documents' on Windows)
#    - Use a relative path (e.g., './my_folder')
folder_path = '.'  # Example: using the current directory

# 2. Call the function to get the list of files.
file_list = get_filenames_from_folder(folder_path)

# 3. Print the results.
if file_list is not None:
    if file_list:
        print(f"Files in '{os.path.abspath(folder_path)}':")
        for filename in file_list:
            print(f"- {filename}")
    else:
        print(f"No files found in '{os.path.abspath(folder_path)}'.")


