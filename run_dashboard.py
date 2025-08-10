"""
Dashboard Startup Script
Run this to start the Streamlit dashboard
"""
import subprocess
import sys
import os

def main():
    """Start the Streamlit dashboard"""
    print("🚀 Starting Portfolio Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔧 Make sure your FastAPI server is running on http://localhost:8000")
    print("-" * 60)
    
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/dashboard.py",
            "--server.port=8502",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Make sure you have installed the requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
