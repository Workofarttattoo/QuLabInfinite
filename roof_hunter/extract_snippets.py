import json
import os

backup_path = "/Users/noone/Library/Containers/com.renfei.SnippetsLab/Data/Library/Application Support/Backups/main 5-4-26, 12-18-54 AM (automatic).snippetslab-backup/library.json"
output_dir = "/Users/noone/QuLabInfinite/roof_hunter/integrations/new_snippets"

os.makedirs(output_dir, exist_ok=True)

with open(backup_path, 'r', encoding='utf-8') as f:
    library = json.load(f)

# Find the folder UUID for "QULAB:Roof Hunter"
folder_uuid = None
folders_available = []
for folder in library.get('folders', []):
    folders_available.append(folder.get('title', ''))
    if "Roof Hunter" in folder.get('title', '') or "RoofHunter" in folder.get('title', ''):
        folder_uuid = folder.get('uuid')
        break

if not folder_uuid:
    print("Could not find 'QULAB:Roof Hunter' folder in the SnippetsLab backup.")
    print("Folders available:", folders_available)
else:
    print(f"Found folder UUID: {folder_uuid}")
    count = 0
    for snippet in library.get('snippets', []):
        if snippet.get('folderUUID') == folder_uuid:
            title = snippet.get('title', 'Untitled').replace('/', '_')
            content = ""
            for fragment in snippet.get('fragments', []):
                content += fragment.get('content', '') + "\n"
            
            # Determine extension
            ext = ".txt"
            if snippet.get('language') == "Python" or "py" in title.lower(): ext = ".py"
            elif snippet.get('language') == "JSON" or "json" in title.lower(): ext = ".json"
            elif snippet.get('language') == "Markdown" or "md" in title.lower(): ext = ".md"
            
            if not title.endswith(ext):
                title += ext
                
            filepath = os.path.join(output_dir, title)
            with open(filepath, 'w', encoding='utf-8') as sf:
                sf.write(content)
            print(f"Extracted: {title}")
            count += 1
            
    print(f"Extracted {count} new snippets to {output_dir}")
