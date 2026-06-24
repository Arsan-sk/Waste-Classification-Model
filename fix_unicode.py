with open('train.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    '\u2550': '=',   # box drawing double horizontal
    '\u2551': '|',   # box drawing double vertical  
    '\u2554': '+',   # box drawing double down and right
    '\u2557': '+',   # box drawing double down and left
    '\u255a': '+',   # box drawing double up and right
    '\u255d': '+',   # box drawing double up and left
    '\u2560': '+',   # box drawing double vertical and right
    '\u2563': '+',   # box drawing double vertical and left
    '\u2500': '-',   # box drawing light horizontal
    '\u2014': '--',  # em dash
    '\u00d7': 'x',   # multiplication sign
    '\u00b1': '+/-', # plus-minus sign
    '\u00b0': ' deg', # degree sign
    '\u2713': '[OK]', # checkmark
    '\u2717': '[X]',  # ballot X
    '\u2610': '[ ]',  # empty checkbox
    '\u2611': '[x]',  # checked checkbox
    '\u2192': '->',   # rightwards arrow (should already be replaced but just in case)
    '\U0001f3af': '(target)',  # target emoji
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('train.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("All non-cp1252 characters replaced successfully")
