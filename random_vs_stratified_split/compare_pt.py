#!/usr/bin/env python3

import argparse
import torch

def compare(a, b, atol, path="root"):
    if type(a) != type(b):
        print(f"{path}: TYPE mismatch ({type(a)} vs {type(b)})")
        return False

    if isinstance(a, dict):
        ok = True
        for k in a:
            if k not in b:
                print(f"{path}: missing key '{k}' in second file")
                ok = False
            else:
                ok &= compare(a[k], b[k], atol, f"{path}.{k}")
        for k in b:
            if k not in a:
                print(f"{path}: extra key '{k}' in second file")
                ok = False
        return ok

    elif isinstance(a, torch.Tensor):
        if not torch.allclose(a, b, atol=atol):
            max_diff = (a - b).abs().max().item()
            print(f"{path}: tensor mismatch (max diff = {max_diff})")
            return False
        return True

    else:
        if a != b:
            print(f"{path}: value mismatch ({a} vs {b})")
            return False
        return True


def main():
    parser = argparse.ArgumentParser(description="Compare two .pt files")
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("--atol", type=float, default=1e-6, help="tolerance for float comparison")
    args = parser.parse_args()

    a = torch.load(args.file1, map_location="cpu")
    b = torch.load(args.file2, map_location="cpu")

    print("Comparing...")
    ok = compare(a, b, args.atol)

    if ok:
        print("✅ Files match")
    else:
        print("❌ Files differ")


if __name__ == "__main__":
    main()