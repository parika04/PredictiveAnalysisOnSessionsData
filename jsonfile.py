import re
import json

INPUT_FILE = 'sessions_export.json'
OUTPUT_FILE = 'sessions_fixed_json.json'

print("Starting conversion from custom log format to valid JSON...")

# --- 1. Read the entire custom log file ---
try:
    with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
        log_text = f.read()
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit()

# --- 2. Define Regex Patterns ---

# Pattern 1: Splits the text into individual session blocks based on the "_id : ..." line.
# We also capture the first session's _id to ensure the split works correctly.
SESSION_SPLIT_PATTERN = r'\n_id\s*:\s*(\w+)'


# Pattern 2: Extracts key-value pairs from lines within a session.
# It handles the keys (e.g., userId), dates, and the complex callStack value.
KEY_VALUE_PATTERN = r'^\s*(\w+)\s*:\s*(.*)'


# --- 3. Split the text into raw session strings ---
# Split the entire text on the occurrence of a new _id.
# The result is a list where each element starts with the _id line.
session_blocks = re.split(SESSION_SPLIT_PATTERN, log_text, flags=re.MULTILINE)

# The first element is usually garbage or empty, and the list alternates between
# the captured _id and the rest of the session block. We combine them.
raw_sessions = []
if len(session_blocks) > 1:
    # session_blocks[0] is garbage. We start iterating from index 1.
    for i in range(1, len(session_blocks), 2):
        # Reconstruct the block: _id value + the rest of the block text
        raw_sessions.append(f"_id : {session_blocks[i]}{session_blocks[i+1]}")
else:
    # If the split didn't work perfectly, assume the whole text is one block (or invalid)
    print("Warning: Could not split logs. Assuming the entire content is one session block.")
    raw_sessions.append(log_text)


# --- 4. Parse each raw session block into a Python dictionary ---
final_json_data = []

for block in raw_sessions:
    # Remove leading/trailing whitespace and ensure the block is not empty
    block = block.strip()
    if not block:
        continue

    current_session = {}
    current_key = None
    
    # Iterate over each line of the block
    for line in block.split('\n'):
        # 1. Try to find a new key/value pair
        match = re.match(KEY_VALUE_PATTERN, line)
        
        if match:
            # Found a new key (e.g., 'userId', 'startedAt')
            current_key = match.group(1).strip()
            current_value = match.group(2).strip()
            
            # Start a new key/value entry
            current_session[current_key] = current_value
            
        elif current_key:
            # 2. If no new key, assume it's a continuation of the previous value (e.g., callStack)
            # Append the cleaned-up line to the existing value.
            if current_key in current_session:
                current_session[current_key] += " " + line.strip()

    # The 'callStack' values (like {[...], [...]}) are still strings, but this is fixed data.
    if current_session:
        final_json_data.append(current_session)

# --- 5. Save the final list of dictionaries as a valid JSON array ---
if final_json_data:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        json.dump(final_json_data, outfile, indent=2)
    
    print(f"\n✅ Conversion successful! {len(final_json_data)} session(s) converted.")
    print(f"The clean JSON data is saved to '{OUTPUT_FILE}'.")
    
    # Optional: Display a small sample of the fixed data
    # print("\n--- Sample of Fixed Data ---")
    # print(json.dumps(final_json_data[0], indent=2))
else:
    print("Failure: No valid session data was extracted.")
    
# Clean up temporary regex objects
del session_blocks, raw_sessions, final_json_data