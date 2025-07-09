#!/usr/bin/env python3
"""
JetX Prediction System Launcher
Simple script to run the enhanced JetX prediction application
"""

import os
import sys
import subprocess

def main():
    """Launch the JetX prediction application"""
    
    # Check if we're in the right directory
    if not os.path.exists("src/main_app.py"):
        print("Error: src/main_app.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import streamlit
        import torch
        import numpy
        import pandas
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install requirements first:")
        print("pip install -r requirements_enhanced.txt")
        sys.exit(1)
    
    # Launch the application
    print("🚀 Launching Enhanced JetX Prediction System...")
    print("📱 Opening in your default browser...")
    print("💡 Tip: Use Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/main_app.py", "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()