# AWS deployment notes (after local verification)

The app is a Streamlit application and can be hosted on an EC2 Linux instance.

1. Create an EC2 instance (Ubuntu LTS; choose a small instance only after checking current AWS pricing and free-tier eligibility).
2. Configure the security group to allow SSH (port 22) only from your IP. Allow Streamlit port 8501 only from the audience IPs for a temporary demo, or put a reverse proxy with HTTPS in front for public use.
3. Clone the repository on the instance and create a Python virtual environment.
4. Install dependencies with `pip install -r requirements.txt`.
5. Confirm `models/best_model.pkl` and `data/processed/used_cars_cleaned.csv` are present.
6. Start the app with `streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501`.
7. For a persistent deployment, configure a systemd service and HTTPS/reverse proxy. Stop or terminate the instance after the demo to avoid ongoing charges.

Do not expose SSH to `0.0.0.0/0`. Do not commit AWS keys, passwords, or Streamlit secrets to GitHub.
