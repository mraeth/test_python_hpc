#!/usr/bin/env python3
"""
Compare complexity of the three implementations.
Metrics: lines of code, number of files, languages used.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class FileStats:
    path: str
    lines: int
    blank: int
    comment: int
    code: int


@dataclass
class Implementation:
    name: str
    directory: str
    files: list[FileStats] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        return sum(f.lines for f in self.files)

    @property
    def total_code(self) -> int:
        return sum(f.code for f in self.files)

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def languages(self) -> set[str]:
        langs = set()
        for f in self.files:
            ext = Path(f.path).suffix
            if ext in ('.cpp', '.hpp', '.h', '.cc'):
                langs.add('C++')
            elif ext == '.py':
                langs.add('Python')
            elif ext in ('.txt', '.cmake'):
                langs.add('CMake')
        return langs


def count_lines(filepath: str) -> FileStats:
    """Count lines, blank lines, comments, and code lines in a file."""
    lines = blank = comment = 0
    in_block_comment = False

    ext = Path(filepath).suffix
    is_python = ext == '.py'
    is_cpp = ext in ('.cpp', '.hpp', '.h', '.cc')
    is_cmake = ext == '.txt' or filepath.endswith('CMakeLists.txt')

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                lines += 1
                stripped = line.strip()

                if not stripped:
                    blank += 1
                    continue

                # C++ block comments
                if is_cpp:
                    if '/*' in stripped and not in_block_comment:
                        in_block_comment = True
                        comment += 1
                        if '*/' in stripped:
                            in_block_comment = False
                        continue
                    if in_block_comment:
                        comment += 1
                        if '*/' in stripped:
                            in_block_comment = False
                        continue
                    if stripped.startswith('//'):
                        comment += 1
                        continue

                # Python comments
                if is_python and stripped.startswith('#'):
                    comment += 1
                    continue

                # CMake comments
                if is_cmake and stripped.startswith('#'):
                    comment += 1
                    continue

    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return FileStats(filepath, 0, 0, 0, 0)

    code = lines - blank - comment
    return FileStats(filepath, lines, blank, comment, code)


def analyze_implementation(name: str, directory: str, patterns: list[str]) -> Implementation:
    """Analyze an implementation directory."""
    impl = Implementation(name, directory)
    base = Path(directory)

    for pattern in patterns:
        for filepath in base.glob(pattern):
            if filepath.is_file():
                stats = count_lines(str(filepath))
                impl.files.append(stats)

    return impl


def main():
    base_dir = Path(__file__).parent

    # Define implementations and their source files
    implementations = [
        analyze_implementation(
            "Kokkos + pybind11",
            str(base_dir / "test_binder"),
            ["src/*.cpp", "src/*.hpp", "CMakeLists.txt"]
        ),
        analyze_implementation(
            "JAX",
            str(base_dir / "test_jax"),
            ["*.py"]
        ),
        analyze_implementation(
            "PyKokkos",
            str(base_dir / "test_pykokkos"),
            ["*.py"]
        ),
    ]

    # Print header
    print("=" * 70)
    print("Complexity Comparison: Lines of Code")
    print("=" * 70)
    print()

    # Summary table
    print(f"{'Implementation':<20} {'Files':>6} {'Total':>8} {'Code':>8} {'Comments':>8} {'Blank':>8} {'Languages'}")
    print("-" * 85)

    for impl in implementations:
        total_blank = sum(f.blank for f in impl.files)
        total_comment = sum(f.comment for f in impl.files)
        langs = ', '.join(sorted(impl.languages))
        print(f"{impl.name:<20} {impl.num_files:>6} {impl.total_lines:>8} {impl.total_code:>8} {total_comment:>8} {total_blank:>8} {langs}")

    print("-" * 85)
    print()

    # Detailed breakdown
    print("Detailed file breakdown:")
    print()

    for impl in implementations:
        print(f"### {impl.name} ({impl.directory})")
        print(f"{'File':<40} {'Lines':>8} {'Code':>8} {'Comments':>8}")
        print("-" * 70)
        for f in sorted(impl.files, key=lambda x: x.path):
            relpath = Path(f.path).name
            print(f"{relpath:<40} {f.lines:>8} {f.code:>8} {f.comment:>8}")
        print()

    # Complexity ratio
    print("=" * 70)
    print("Complexity Ratio (relative to simplest)")
    print("=" * 70)

    min_code = min(impl.total_code for impl in implementations)
    for impl in implementations:
        ratio = impl.total_code / min_code if min_code > 0 else 0
        bar = "█" * int(ratio * 10)
        print(f"{impl.name:<20} {impl.total_code:>4} lines  {ratio:>5.1f}x  {bar}")

    print()
    print("Note: Kokkos requires C++ and build configuration, but offers")
    print("      maximum performance and control. JAX/PyKokkos are pure Python.")


if __name__ == "__main__":
    main()
