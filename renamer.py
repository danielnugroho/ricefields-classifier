# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:27:31 2024

@author: Daniel.Nugroho
"""

import os
import re
from datetime import datetime

# Function to extract date from filename
def extract_date(filename):
    match = re.search(r'\d{2}[A-Za-z]{3}\d{4}', filename)
    matchvh = re.search('VH', filename)
    
    if match:
        if matchvh:
            return datetime.strptime(match.group(), '%d%b%Y').strftime('%Y%m%d') + "_VH"
        else:
            return datetime.strptime(match.group(), '%d%b%Y').strftime('%Y%m%d') + "_VV"
    else:
        return None

# Folder path
folder_path = "B4-2022"

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        # Extract date from filename
        date = extract_date(filename)
        if date:
            # Create new filename
            new_filename = date + os.path.splitext(filename)[1]
            # Rename the file
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
