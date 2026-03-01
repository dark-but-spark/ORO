from pathlib import Path

def select_best_test(files, preferred_exts):
    if not files:
        return []
    if len(files) == 1:
        return [files[0]]
    
    # Explicitly prioritize 4ch files first
    ch4_files = [f for f in files if '_4ch' in f.name.lower()]
    print(f'Found {len(ch4_files)} 4ch files')
    if ch4_files:
        # Among 4ch files, sort by preferred extensions
        pref = [e.lower() for e in preferred_exts]
        ch4_files.sort(key=lambda p: pref.index(p.suffix.lower()) if p.suffix.lower() in pref else len(pref))
        print(f'Selected 4ch file: {ch4_files[0].name}')
        return [ch4_files[0]]
    
    # If no 4ch files, use original logic for regular files
    pref = [e.lower() for e in preferred_exts]
    files_sorted = sorted(files, key=lambda p: pref.index(p.suffix.lower()) if p.suffix.lower() in pref else len(pref))
    print(f'Selected regular file: {files_sorted[0].name}')
    return [files_sorted[0]]

# Test with actual files
files = list(Path('data/masks').glob('102_1_6_44_jpg.rf.008d7a4772e4bf8d621bd815723ae428*'))
preferred_exts = ['.npz', '.pt', '.pth', '.tiff', '.tif', '.png', '.jpg', '.jpeg']

print('Testing select_best function:')
result = select_best_test(files, preferred_exts)
print('Result:', [f.name for f in result])
