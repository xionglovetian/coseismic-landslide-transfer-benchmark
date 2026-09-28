"""Compare two checkpoint state dictionaries tensor-by-tensor."""

from __future__ import annotations

import argparse
import hashlib
import sys

import torch


def canonical_state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    left_state = left.get("state_dict", left)
    right_state = right.get("state_dict", right)
    if list(left_state) != list(right_state):
        print("FAIL: state-dict keys differ")
        return 1
    mismatched = []
    different = 0
    total = 0
    max_abs_diff = 0.0
    for key in left_state:
        a = left_state[key]
        b = right_state[key]
        if a.shape != b.shape:
            mismatched.append(key)
            continue
        total += a.numel()
        different += int((a != b).sum().item())
        if a.is_floating_point():
            max_abs_diff = max(max_abs_diff, float((a.float() - b.float()).abs().max().item()))
    if mismatched or different:
        print(f"FAIL: mismatched tensors={len(mismatched)} different_elements={different}/{total}")
        return 1
    print(
        f"PASS: {len(left_state)} tensors, {total} elements, "
        f"0 different elements, max_abs_diff={max_abs_diff}"
    )
    print(f"canonical_state_sha256={canonical_state_hash(left_state)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())