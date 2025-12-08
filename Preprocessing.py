import pandas as pd
import json

# Define the file paths
INPUT_FILE = 'sessions_fixed_json.json' 

# --- Load the Data ---
with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

df_sessions = pd.DataFrame(data)

print(f"Loaded {len(df_sessions)} session records.")
print(f"Success: Loaded {len(df_sessions)} session records.")

print("DataFrame Head (First 3 rows):")

print(df_sessions.head(3))

# We use the pd.to_datetime() function to convert the columns.
# It automatically detects the ISO 8601 format of your timestamps.
df_sessions['startedAt'] = pd.to_datetime(df_sessions['startedAt'])
df_sessions['endedAt'] = pd.to_datetime(df_sessions['endedAt'])

# Display the data types to verify the conversion
print("\nDataFrame dtypes after conversion:")
print(df_sessions.dtypes)
print(df_sessions.head(3))
# Preprocessing to convert the string 'callStack': "{...}" into a list []
# We do this before using json_normalize.

processed_data = []

for session in data:
    call_stack_str = session.get('callStack', '{}')
    
    # Handle empty/null call stacks: Convert to an empty list
    if not call_stack_str or call_stack_str in ('{}', 'null'):
        session['callStack'] = []
    
    # Handle non-empty string call stacks like '{[log1], [log2], ...}'
    elif call_stack_str.startswith('{') and call_stack_str.endswith('}'):
        content = call_stack_str[1:-1].strip()
        
        # Split by the separator '], [' which is common in your logs
        # This creates the list structure needed for pd.json_normalize
        log_entries = [entry.strip() for entry in content.split('], [')]
        
        # Reconstruct the list of log strings, ensuring all brackets are present
        fixed_log_entries = []
        for i, entry in enumerate(log_entries):
            # Ensure the opening bracket is there for the first element
            if i == 0 and not entry.startswith('['):
                entry = '[' + entry
            # Ensure the closing bracket is there for all elements except the last
            if i < len(log_entries) - 1 and not entry.endswith(']'):
                entry = entry + ']'
            
            # The last element will be correctly terminated when content is well-formed

            fixed_log_entries.append(entry)

        session['callStack'] = fixed_log_entries
    
    # Append the session (now with a correct callStack type) to the list
    processed_data.append(session)
print(f"\nPreprocessing Complete: Processed {len(processed_data)} sessions.")
print("\nSample Processed Session (First 1):")
print(processed_data[0])
print(processed_data[1])

# --- Flatten Data into the CSV Structure ---

# This creates the foundation of your CSV data by exploding the callStack
df_flat = pd.json_normalize(
    processed_data,
    record_path=['callStack'],
    meta=['_id', 'userId', 'startedAt', 'endedAt', 'session_duration_sec'],
    meta_prefix='session_',
    record_prefix='callStack_',
    errors='ignore'
)

# Rename the new column containing the raw log entry (which will be parsed later)
df_flat.rename(columns={'callStack_0': 'callStack_log_raw'}, inplace=True)

print(f"\nData Flattening Complete: DataFrame has {len(df_flat)} total API call rows.")
print("\nDataFrame Head (Flattened):")
print(df_flat.head())

# --- SAVE TO CSV ---
OUTPUT_CSV = 'sessions_flattened_ready.csv'
df_flat.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Data converted to CSV! File saved as {OUTPUT_CSV}")