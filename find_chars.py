with open('train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

found = False
for i, line in enumerate(lines):
    for c in line:
        if ord(c) > 127:
            try:
                c.encode('cp1252')
            except UnicodeEncodeError:
                found = True
                # Write to file to avoid console encoding issues
                with open('unicode_issues.txt', 'a') as out:
                    out.write(f"Line {i+1}: char U+{ord(c):04X} in: {line.rstrip()}\n")

if not found:
    with open('unicode_issues.txt', 'w') as out:
        out.write("NO ISSUES FOUND - all characters are cp1252 compatible\n")
