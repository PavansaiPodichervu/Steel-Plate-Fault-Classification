
import subprocess
import sys

scripts = [
    "data_preprocessing.py",
    "train_models.py",
    "evaluate_models.py"
]

for script in scripts:
    print(f"\nExecuting {script}...")

    subprocess.run(
        [sys.executable, script],
        check=True
    )

print("\nProject execution completed!")
print("Launching Streamlit application...")

subprocess.run([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py"
])
