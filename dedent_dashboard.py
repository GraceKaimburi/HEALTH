from pathlib import Path
path = Path('app.py')
text = path.read_text(encoding='utf-8')
lines = text.splitlines(True)
start = None
for i, ln in enumerate(lines):
    if '# Define dashboard layout columns before use' in ln:
        start = i
        break
if start is None:
    raise SystemError('marker not found')
for j in range(start+1, len(lines)):
    if lines[j].startswith('        '):
        lines[j] = lines[j][4:]
path.write_text(''.join(lines), encoding='utf-8')
print('done')