import os

# Corrected paths: go up one directory from utils, then into datastore
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "datastore", "generations.db")
IMAGES_DIR = os.path.join(BASE_DIR, "datastore", "images")
MODELS_DIR = os.path.join(BASE_DIR, "datastore", "models")

def delete_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Database deleted.")
    else:
        print("Database file not found.")

def delete_folder_contents(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print(f"All files deleted in {folder}")
    else:
        print(f"Folder not found: {folder}")

def cleanup_all():
    delete_database()
    delete_folder_contents(IMAGES_DIR)
    delete_folder_contents(MODELS_DIR)

if __name__ == "__main__":
    cleanup_all()