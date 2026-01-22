import json
import sys

files = sys.argv[1:]
for file in files:
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error in file {file} at line {i+1}: {e}, {line.strip()}")
        print(f"Checked file {file} done")
