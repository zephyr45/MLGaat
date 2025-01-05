import os
import gdown
import subprocess

# Function to download a file from Google Drive
def download_file_from_google_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Google Drive file IDs for similarity files
pickle_files_google_drive = {
    'similarity_Vectorization.pkl': '1NSMACB1wgPdOf6ZjDb7GcObhUh614Urs',
    'similarity_LSA.pkl': '17XhT43YiR8yCWe6NTLhwQz9iVeNkb2-2'
}

# Download similarity files from Google Drive
for file_name, file_id in pickle_files_google_drive.items():
    try:
        if not os.path.exists(file_name):  # Check if file is already downloaded
            download_file_from_google_drive(file_id, file_name)
        else:
            print(f"{file_name} already exists, skipping download.")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")

# Run the Streamlit app (app.py) after ensuring necessary files are downloaded
subprocess.run(["streamlit", "run", "app.py"])
