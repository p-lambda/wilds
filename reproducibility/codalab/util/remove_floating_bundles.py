import argparse
import subprocess


"""
Removes your floating bundles in CodaLab.

Example usage:

    python3 reproducibility/codalab/util/remove_floating_bundles.py

"""


def run(command):
    def clean_output(output):
        # Clean output to get UUID, states, etc.
        return output.strip("\n").strip()

    # Print the command that is about to be run
    print(" ".join(command))

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    # Print the stdout of the command
    print(process.stdout)
    return clean_output(process.stdout)


def main():
    run(["cl", "work", "https://worksheets.codalab.org::"])

    # Search for your own floating bundles
    floating_bundles = run(
        ["cl", "search", ".floating", ".mine", ".limit=200", "--uuid-only"]
    ).split("\n")

    for bundle in floating_bundles:
        if bundle:
            run(["cl", "rm", bundle, "--force"])
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove floating bundles")

    # Parse args and run this script
    main()
