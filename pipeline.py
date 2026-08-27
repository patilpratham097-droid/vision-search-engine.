import subprocess
import os

def run_pipeline():
    print("--- Starting Vision Search Pipeline ---")
    
    #  Run the Python Feature Extractor
    print("\n[Step 1] Extracting image features...")
    # We use subprocess to run the extract script just like we would in the terminal
    extract_result = subprocess.run(["python", "extract.py"], capture_output=True, text=True)
    
    if extract_result.returncode != 0:
        print("Error running extraction:")
        print(extract_result.stderr)
        return

    print(extract_result.stdout)

    # 2 Check if the C++ executable exists
    cpp_executable = "./search_engine" if os.name != "nt" else "search_engine.exe"
    
    if not os.path.exists(cpp_executable):
        print(f"Error: {cpp_executable} not found. Did you compile your C++ code?")
        return

    #  Run the C++ Search Engine
    print("\n[Step 2] Searching database...")
    search_result = subprocess.run([cpp_executable], capture_output=True, text=True)
    
    if search_result.returncode != 0:
        print("Error running search engine:")
        print(search_result.stderr)
        return
        
    print(search_result.stdout)
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    run_pipeline()