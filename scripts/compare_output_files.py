import sys
from collections import defaultdict


def hash_line(line):
    parts = line.split()
    return hash((parts[0], parts[1], float(parts[2])))


def find_unmatched_lines(file1, file2):
    lines1 = defaultdict(list)
    lines2 = set()

    print(f"Reading {file1}...")
    with open(file1, 'r') as f1:
        for i, line in enumerate(f1):
            line_hash = hash_line(line.strip())
            lines1[line_hash].append((i, line.strip()))

    print(f"Reading {file2}...")
    with open(file2, 'r') as f2:
        for line in f2:
            lines2.add(hash_line(line.strip()))

    unmatched = []
    for line_hash, lines in lines1.items():
        if line_hash not in lines2:
            unmatched.extend(lines)

    if unmatched:
        print(f"Lines in {file1} that are not in {file2}:")
        for i, line in sorted(unmatched):
            print(f"Line {i + 1}: {line}")
    else:
        print(f"All lines in {file1} are present in {file2}.")

    return len(unmatched), len(lines1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1> <file2>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    print("Comparing files...")
    unmatched1, lines_in_file1 = find_unmatched_lines(file1, file2)
    print("\n" + "=" * 50 + "\n")
    unmatched2, lines_in_file2 = find_unmatched_lines(file2, file1)

    percent_diff1 = (unmatched1 / lines_in_file1) * 100 if lines_in_file1 > 0 else 0
    percent_diff2 = (unmatched2 / lines_in_file2) * 100 if lines_in_file2 > 0 else 0

    print("\nSummary:")
    print(f"{file1} has {lines_in_file1} lines.")
    print(f"{file2} has {lines_in_file2} lines.")
    print(f"Lines in {file1} not in {file2}: {unmatched1} ({percent_diff1:.2f}%)")
    print(f"Lines in {file2} not in {file1}: {unmatched2} ({percent_diff2:.2f}%)")
