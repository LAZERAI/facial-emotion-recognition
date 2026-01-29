#!/usr/bin/env python3
"""
Smoke check script: Verify environment and required files.
Run this before training/inference to catch issues early.
"""

import os
import sys

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. You have:", sys.version)
        return False
    print("✅ Python version:", sys.version.split()[0])
    return True

def check_imports():
    required = ["tensorflow", "deepface", "cv2", "numpy", "sklearn", "PIL"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
            print(f"✅ {mod}")
        except ImportError:
            missing.append(mod)
            print(f"❌ {mod} (missing)")
    return len(missing) == 0

def check_files():
    files = [
        "emotion__images",
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    ]
    for f in files:
        if os.path.exists(f):
            print(f"✅ {f}")
        else:
            print(f"⚠️  {f} (missing - will use fallback)")
    return True  # Not critical

def main():
    print("🔍 Smoke Check for Facial Emotion Recognition\n")
    ok = True
    ok &= check_python_version()
    ok &= check_imports()
    check_files()
    print("\n" + ("✅ All checks passed!" if ok else "❌ Some issues found. Fix before proceeding."))

if __name__ == "__main__":
    main()