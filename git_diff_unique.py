import subprocess
import os
import sys
import tempfile
import argparse
import PIL.Image
import imagehash


def run_git_command(cmd: list[str]) -> str:
    """
    Run a Git command and return its output as a string.

    Parameters
    ----------
    cmd : list[str]
        The Git command and arguments to run.

    Returns
    -------
    str
        The standard output of the Git command.

    Raises
    ------
    RuntimeError
        If the Git command fails.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def get_changed_images(git_range: str) -> list[str]:
    """
    Get a list of image files changed in a given Git range.

    Parameters
    ----------
    git_range : str
        Git commit range, e.g., 'origin/main..HEAD'.

    Returns
    -------
    list[str]
        List of changed image file paths filtered by common image extensions.
    """
    print(f"Getting changed images in range '{git_range}'...")
    cmd = ['git', 'diff', '--name-only', git_range]
    output = run_git_command(cmd)
    files = [f for f in output.splitlines() if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    print(f"Found {len(files)} changed image files.")
    return files


def get_commit_before_range(git_range: str) -> str:
    """
    Get the commit hash at the start of the Git range.

    Parameters
    ----------
    git_range : str
        Git commit range, e.g., 'origin/main..HEAD'.

    Returns
    -------
    str
        Commit hash of the start of the range.
    """
    commit = run_git_command(['git', 'rev-parse', git_range.split('..')[0]])
    print(f"Base commit for comparison: {commit}")
    return commit


def checkout_tree(commit: str, destination: str) -> None:
    """
    Clone the current repository and check out a specific commit into a temporary directory.

    Parameters
    ----------
    commit : str
        The commit hash to check out.
    destination : str
        The path of the temporary directory where the commit will be checked out.
    """
    print(f"Cloning repo and checking out commit {commit} to temporary directory...")
    subprocess.run(['git', 'clone', '.', destination, '--quiet', '--no-checkout'], check=True)
    subprocess.run(['git', '-C', destination, 'checkout', commit, '--', '.'], check=True)
    print("Checkout done.")


def compute_hash(image_path: str, hashfunc=imagehash.phash) -> any:
    """
    Compute the perceptual hash of an image file.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    hashfunc : callable, optional
        Hash function from imagehash library (default is phash).

    Returns
    -------
    imagehash.ImageHash or None
        The computed perceptual hash or None if hashing fails.
    """
    try:
        img = PIL.Image.open(image_path)
        return hashfunc(img)
    except Exception as e:
        print(f"  Error hashing image '{image_path}': {e}")
        return None


def group_similar_changes(images_info: list[tuple[str, any, any, int]], threshold: int = 5) -> list[list[str]]:
    """
    Group images that are similar based on old, new, and delta perceptual hashes.

    Parameters
    ----------
    images_info : list[tuple[str, imagehash.ImageHash, imagehash.ImageHash, int]]
        List of tuples containing:
        (image path, old hash, new hash, delta hash difference).
    threshold : int, optional
        Maximum Hamming distance allowed to consider images similar (default is 5).

    Returns
    -------
    list[list[str]]
        List of groups, each group is a list of image file paths.
    """
    print("Grouping images based on similarity...")
    groups = []
    used = set()

    for i, (path1, old1, new1, delta1) in enumerate(images_info):
        if path1 in used:
            continue
        group = [path1]
        used.add(path1)
        for j in range(i + 1, len(images_info)):
            path2, old2, new2, delta2 = images_info[j]
            if path2 in used:
                continue
            if (abs(old1 - old2) <= threshold and
                abs(new1 - new2) <= threshold and
                abs(delta1 - delta2) <= threshold):
                group.append(path2)
                used.add(path2)
        groups.append(group)

    print(f"Grouping done. {len(groups)} groups formed.")
    return groups


def analyze_git_image_changes(git_range: str, threshold: int) -> dict[str, any]:
    """
    Analyze image changes between two Git commits, grouping visually similar changed images.

    Parameters
    ----------
    git_range : str
        Git commit range, e.g., 'origin/main..HEAD'.
    threshold : int
        Hamming distance threshold to consider images similar.

    Returns
    -------
    dict[str, any]
        Dictionary containing:
        - total_changed: int, number of changed images with valid hashes
        - unique_change_groups: int, number of unique change groups
        - groups: list[list[str]], groups of image paths
        - unhashable: list[tuple[str, str]], images missing old/new file or with failed hash and reason
    """
    changed_files = get_changed_images(git_range)
    base_commit = get_commit_before_range(git_range)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkout_tree(base_commit, tmp_dir)

        images_info = []
        unhashable_images = []
        print(f"Processing {len(changed_files)} changed images...")
        for idx, rel_path in enumerate(changed_files, 1):
            old_path = os.path.join(tmp_dir, rel_path)
            new_path = rel_path

            if not os.path.exists(new_path):
                unhashable_images.append((rel_path, "New file missing"))
                print(f"[{idx}/{len(changed_files)}] Skipping '{rel_path}' (new file missing)")
                continue
            if not os.path.exists(old_path):
                unhashable_images.append((rel_path, "Old file missing"))
                print(f"[{idx}/{len(changed_files)}] Skipping '{rel_path}' (old file missing)")
                continue

            print(f"[{idx}/{len(changed_files)}] Hashing '{rel_path}'...")
            old_hash = compute_hash(old_path)
            new_hash = compute_hash(new_path)

            if old_hash is None:
                unhashable_images.append((rel_path, "Failed to hash old file"))
                print(f"  Warning: Could not hash old '{rel_path}'.")
                continue
            if new_hash is None:
                unhashable_images.append((rel_path, "Failed to hash new file"))
                print(f"  Warning: Could not hash new '{rel_path}'.")
                continue

            delta = new_hash - old_hash
            images_info.append((rel_path, old_hash, new_hash, delta))

        print(f"Hashing done. {len(images_info)} images with valid hashes.")
        groups = group_similar_changes(images_info, threshold=threshold)

    return {
        "total_changed": len(images_info),
        "unique_change_groups": len(groups),
        "groups": groups,
        "unhashable": unhashable_images
    }


def main() -> None:
    """
    Main entry point of the script. Parses arguments, runs analysis, and prints results.
    """
    parser = argparse.ArgumentParser(description="Group visually similar image changes in a Git diff.")
    parser.add_argument("git_range", help="Git commit range, e.g. origin/main..HEAD")
    parser.add_argument("--threshold", type=int, default=5, help="Hamming distance threshold (default: 5)")
    args = parser.parse_args()

    results = analyze_git_image_changes(args.git_range, args.threshold)

    print()
    print(f"Changed image files in range '{args.git_range}': {results['total_changed']}")
    print(f"Unique change groups: {results['unique_change_groups']}")
    print(f"Images missing or unhashable: {len(results['unhashable'])}")
    print()

    for i, group in enumerate(results["groups"], 1):
        print(f"Group {i}: {len(group)} images")
        example_file = group[0]
        print(f"  Example: {example_file}")
        try:
            diff = run_git_command(['git', 'diff', args.git_range, '--', example_file])
            print(f"  Git diff output for {example_file}:\n{'-'*60}\n{diff}\n{'-'*60}\n")
        except Exception as e:
            print(f"  Failed to diff {example_file}: {e}")

    if results["unhashable"]:
        for img, reason in results["unhashable"]:
            print(f"  {img} -> {reason}")
            try:
                diff = run_git_command(['git', 'diff', args.git_range, '--', img])
                print(f"  Git diff output for {img}:\n{'-'*60}\n{diff}\n{'-'*60}\n")
            except Exception as e:
                print(f"  Failed to diff {img}: {e}")


if __name__ == "__main__":
    main()
