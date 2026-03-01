from pathlib import Path
files = list(Path('data/masks').glob('102_1_6_44_jpg.rf.008d7a4772e4bf8d621bd815723ae428*'))
print('Files:')
for f in files:
    print(f.name)

print()
print('Testing priority logic:')
pref_dict = {'.npz': 0, '.pt': 1, '.pth': 2, '.tiff': 3, '.tif': 4, '.png': 5, '.jpg': 6, '.jpeg': 7}

def get_priority(filepath):
    filename = filepath.name.lower()
    print(f'Testing: {filename}')
    
    # Check for 4ch variants first
    if '_4ch.' in filename:
        base_ext = '.' + filename.split('_4ch.')[-1]
        print(f'  Found _4ch pattern, base_ext={base_ext}')
        if base_ext in pref_dict:
            priority = pref_dict[base_ext] - 0.5  # Higher priority
            print(f'  Priority: {priority}')
            return priority
        else:
            print(f'  Unknown 4ch variant, priority: {len(pref_dict)}')
            return len(pref_dict)
    # Regular files
    elif filepath.suffix.lower() in pref_dict:
        priority = pref_dict[filepath.suffix.lower()]
        print(f'  Regular file, priority: {priority}')
        return priority
    else:
        print(f'  Unknown extension, priority: {len(pref_dict)}')
        return len(pref_dict)

print()
priorities = [(f.name, get_priority(f)) for f in files]
print()
print('Sorted by priority:')
sorted_files = sorted(files, key=get_priority)
for f in sorted_files:
    print(f.name)
