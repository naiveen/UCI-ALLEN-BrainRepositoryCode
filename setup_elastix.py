import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile

def download_file(url, dest):
    print(f"Downloading {url} to {dest}")
    urllib.request.urlretrieve(url, dest)

def extract_zip(src, dest):
    print(f"Extracting {src} to {dest}")
    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(dest)

def execute_command(command):
    print(f"Executing command: {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode == 0:
        print(f"Output: {result.stdout}")
    else:
        print(f"Error: {result.stderr}")

def remove_directory(path):
    if os.path.exists(path):
        print(f"Removing existing directory {path}")
        shutil.rmtree(path)

def set_env_variable_persistent(name, value):
    if platform.system() == "Windows":
        command = f'setx {name} "{value}"'
        execute_command(command)
    else:
        shell_profile = os.path.expanduser("~/.bashrc")
        if os.path.exists(os.path.expanduser("~/.zshrc")):
            shell_profile = os.path.expanduser("~/.zshrc")
        with open(shell_profile, 'a') as file:
            file.write(f'\nexport {name}="{value}"\n')
        print(f"Environment variable {name} set in {shell_profile}. Please restart your terminal or run `source {shell_profile}` to apply the changes.")

# Define the variables
url = "https://github.com/SuperElastix/elastix/releases/download/5.1.0/elastix-5.1.0-win64.zip"
registration_home = os.getcwd()
zip_file = os.path.join(registration_home, "elastix-5.1.0-win64.zip")
elastix_home = os.path.join(registration_home, "elastix")
extract_dest = elastix_home

# Remove elastix_home if it exists
remove_directory(elastix_home)

# Download the file
download_file(url, zip_file)

# Extract the file
extract_zip(zip_file, extract_dest)

# Remove the zip file
os.remove(zip_file)


# Set the environment variable persistently
set_env_variable_persistent("ELASTIX_HOME", elastix_home)

# Print the set environment variable
print(f"ELASTIX_HOME is set to {elastix_home}. Please restart your terminal or run the appropriate command to apply the changes.")
