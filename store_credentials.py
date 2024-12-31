import os
from getpass import getpass


ENV_PATH = '.env'
GITIGNORE_PATH = '.gitignore'

def verify_env_gitignore():
    # Add .env to .gitignore if not already present
    if not os.path.exists(GITIGNORE_PATH):
        with open(".gitignore", "w") as gitignore_file:
            gitignore_file.write(".env\n")
    else:
        with open(".gitignore", "r+") as gitignore_file:
            if ".env" not in gitignore_file.read():
                gitignore_file.write(".env\n")

def store_db_password():
    # Prompt the user for credentials
    db_user = input("Enter database username: ")
    db_password = getpass("Enter database password: ")  # Hide password input
    # api_key = getpass("Enter API key: ")  # Hide API key input
    with open(ENV_PATH, "a") as env_file:
        env_file.write(f"DB_USER={db_user}\n")
        env_file.write(f"DB_PASSWORD={db_password}\n")
    print("Credentials have been securely stored in the .env file.")

def verify_env_file():
    # Write credentials to .env file
    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, 'w').close()


def main():
    # verify env file exists or create if necessary
    verify_env_file()
    # wanna make sure we don't push .env to remote repositories
    verify_env_gitignore()
    store_db_password()
    

if __name__ == "__main__":
    main()