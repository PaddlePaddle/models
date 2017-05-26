import sys

path = sys.argv[1]

lines = []

with open(path) as f:
    code_block_start = False
    code_block_end = False
    in_block = False
    code_block_start_flag = False
    block_end = False

    for line in f:
        line = line.rstrip()
        real_block_start = line[:4] == ' ' * 4
        block_start = real_block_start or not line

        if real_block_start and not code_block_start_flag:
            code_block_start_flag = True
            lines.append('```python')

        if code_block_start_flag and block_start:
            line = line[4:]

        if code_block_start_flag and not block_start:
            lines.append('```')
            code_block_start_flag = False
            lines.append('')

        lines.append(line)

print '\n'.join(lines)
